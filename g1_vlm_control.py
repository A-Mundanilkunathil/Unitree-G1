"""
Vision Language Model (VLM) Integration for Unitree G1
Allows robot to decide actions based on visual input and natural language instructions
Uses Google Gemini 2.5 Flash
"""

import cv2
import numpy as np
import time
import json
from typing import List, Dict, Optional, Tuple
from PIL import Image
import google.generativeai as genai


class VLMActionPlanner:
    """
    VLM-based action planner for Unitree G1
    Uses Google Gemini 2.5 Flash model
    """
    
    def __init__(self, api_key=None, model_name='gemini-2.5-flash'):
        """
        Initialize VLM action planner with Google Gemini
        
        Args:
            api_key: Google API key (get from https://makersuite.google.com/app/apikey)
            model_name: Gemini model name (default: gemini-2.5-flash)
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Available robot actions
        self.available_actions = {
            'stand_up': 'Stand up from sitting/lying position',
            'stand_down': 'Sit down or lie down',
            'walk_forward': 'Walk forward (speed: 0.1-0.5 m/s)',
            'walk_backward': 'Walk backward (speed: 0.1-0.5 m/s)',
            'turn_left': 'Turn left (speed: 0.1-0.8 rad/s)',
            'turn_right': 'Turn right (speed: 0.1-0.8 rad/s)',
            'strafe_left': 'Move sideways to the left',
            'strafe_right': 'Move sideways to the right',
            'stop': 'Stop all movement',
            'track_object': 'Track and follow a specific object',
            'look_around': 'Rotate to scan the environment',
            'approach_object': 'Move closer to an object',
            'avoid_obstacle': 'Move away from obstacles',
            'wave': 'Wave with arm (if equipped)',
            'none': 'Do nothing, stay in current state'
        }
        
        # Initialize Gemini client
        if not self.api_key:
            raise ValueError("Google API key is required. Get one from: https://makersuite.google.com/app/apikey")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        print(f"✓ VLM Action Planner initialized (Google Gemini: {self.model_name})")
    
    def prepare_image(self, image):
        """
        Prepare image for Gemini API
        
        Args:
            image: numpy array (OpenCV format) or PIL Image
            
        Returns:
            PIL Image ready for Gemini
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Resize if too large (Gemini can handle up to 4MB)
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return image
    
    def decide_action(self, 
                     image, 
                     instruction: str,
                     context: Optional[str] = None,
                     previous_actions: Optional[List[str]] = None) -> Dict:
        """
        Use VLM to decide robot action based on visual input and instruction
        
        Args:
            image: Current camera frame (numpy array or PIL Image)
            instruction: User instruction (e.g., "find and approach the chair")
            context: Additional context about environment or task
            previous_actions: List of recent actions taken
            
        Returns:
            Dictionary with:
                - action: Action to take
                - parameters: Parameters for the action
                - reasoning: VLM's reasoning
                - confidence: Confidence score (0-1)
        """
        # Build prompt
        prompt = self._build_action_prompt(instruction, context, previous_actions)
        
        # Get VLM response
        response = self._query_vlm(image, prompt)
        
        # Parse response
        action_plan = self._parse_vlm_response(response)
        
        return action_plan
    
    def _build_action_prompt(self, 
                            instruction: str,
                            context: Optional[str] = None,
                            previous_actions: Optional[List[str]] = None) -> str:
        """Build prompt for VLM"""
        
        # List available actions
        actions_list = "\n".join([f"- {action}: {desc}" 
                                 for action, desc in self.available_actions.items()])
        
        prompt = f"""You are controlling a Unitree G1 humanoid robot. Based on the image and instruction, decide the next action.

INSTRUCTION: {instruction}

AVAILABLE ACTIONS:
{actions_list}

"""
        
        if context:
            prompt += f"CONTEXT: {context}\n\n"
        
        if previous_actions:
            prompt += f"PREVIOUS ACTIONS: {', '.join(previous_actions[-5:])}\n\n"
        
        prompt += """Please analyze the image and respond in JSON format:
{
  "action": "action_name",
  "parameters": {
    "speed": 0.3,
    "duration": 2.0,
    "target": "object_name"
  },
  "reasoning": "Why this action is appropriate",
  "confidence": 0.95
}

Choose the most appropriate action based on:
1. What you see in the image
2. The user's instruction
3. Safety (avoid dangerous actions)
4. Task progress (consider previous actions)

Respond ONLY with valid JSON, no other text."""
        
        return prompt
    
    def _query_vlm(self, image, prompt: str) -> str:
        """Query Google Gemini API"""
        try:
            # Prepare image for Gemini
            pil_image = self.prepare_image(image)
            
            # Generate content with Gemini
            response = self.model.generate_content(
                [prompt, pil_image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500,
                )
            )
            
            return response.text
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Fallback response if VLM fails"""
        return json.dumps({
            "action": "stop",
            "parameters": {},
            "reasoning": "VLM unavailable, defaulting to safe stop action",
            "confidence": 0.5
        })
    
    def _parse_vlm_response(self, response: str) -> Dict:
        """Parse VLM response into action plan"""
        try:
            # Extract JSON from response (in case there's extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                action_plan = json.loads(json_str)
                
                # Validate action
                if action_plan.get('action') not in self.available_actions:
                    print(f"⚠️  Invalid action: {action_plan.get('action')}, using 'stop'")
                    action_plan['action'] = 'stop'
                
                # Ensure required fields
                if 'parameters' not in action_plan:
                    action_plan['parameters'] = {}
                if 'reasoning' not in action_plan:
                    action_plan['reasoning'] = "No reasoning provided"
                if 'confidence' not in action_plan:
                    action_plan['confidence'] = 0.5
                
                return action_plan
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Error parsing VLM response: {e}")
            return json.loads(self._get_fallback_response())
    
    def get_action_description(self, action: str) -> str:
        """Get description of an action"""
        return self.available_actions.get(action, "Unknown action")
    
    def validate_action(self, action: str, parameters: Dict) -> Tuple[bool, str]:
        """
        Validate action and parameters for safety
        
        Returns:
            (is_valid, message)
        """
        if action not in self.available_actions:
            return False, f"Unknown action: {action}"
        
        # Check speed limits
        if 'speed' in parameters:
            speed = parameters['speed']
            if speed < 0 or speed > 1.0:
                return False, f"Speed {speed} out of range [0, 1.0]"
        
        # Check duration limits
        if 'duration' in parameters:
            duration = parameters['duration']
            if duration < 0 or duration > 30:
                return False, f"Duration {duration} out of range [0, 30]"
        
        return True, "Action validated"


class G1VLMController:
    """
    High-level controller that integrates VLM with robot control
    """
    
    def __init__(self, robot_controller, vision_system, vlm_planner: VLMActionPlanner):
        """
        Initialize VLM controller
        
        Args:
            robot_controller: Robot movement controller (e.g., G1HighLevelController)
            vision_system: Vision detection system (e.g., G1VisionDetector)
            vlm_planner: VLM action planner
        """
        self.robot = robot_controller
        self.vision = vision_system
        self.vlm = vlm_planner
        
        self.action_history = []
        self.max_history = 20
        
        print("✓ G1 VLM Controller initialized")
    
    def execute_instruction(self, 
                           instruction: str,
                           max_actions: int = 10,
                           action_interval: float = 2.0) -> List[Dict]:
        """
        Execute high-level instruction using VLM
        
        Args:
            instruction: Natural language instruction
            max_actions: Maximum number of actions to take
            action_interval: Time between actions (seconds)
            
        Returns:
            List of executed actions
        """
        print(f"\n{'='*60}")
        print(f"EXECUTING: {instruction}")
        print(f"{'='*60}\n")
        
        executed_actions = []
        
        for i in range(max_actions):
            print(f"\n--- Action {i+1}/{max_actions} ---")
            
            # Get current frame
            with self.vision.frame_lock:
                if self.vision.frame is None:
                    print("⚠️  No camera frame available")
                    break
                frame = self.vision.frame.copy()
            
            # Decide action using VLM
            print("Consulting VLM...")
            action_plan = self.vlm.decide_action(
                image=frame,
                instruction=instruction,
                previous_actions=[a['action'] for a in self.action_history[-5:]]
            )
            
            print(f"\nVLM Decision:")
            print(f"  Action: {action_plan['action']}")
            print(f"  Reasoning: {action_plan['reasoning']}")
            print(f"  Confidence: {action_plan['confidence']:.2f}")
            
            # Validate action
            is_valid, msg = self.vlm.validate_action(
                action_plan['action'],
                action_plan['parameters']
            )
            
            if not is_valid:
                print(f"⚠️  Action validation failed: {msg}")
                break
            
            # Execute action
            success = self._execute_action(action_plan)
            
            action_plan['success'] = success
            action_plan['timestamp'] = time.time()
            executed_actions.append(action_plan)
            
            # Update history
            self.action_history.append(action_plan)
            if len(self.action_history) > self.max_history:
                self.action_history.pop(0)
            
            # Check if task is complete
            if action_plan['action'] == 'none':
                print("\n✓ Task completed (VLM decided no further action needed)")
                break
            
            # Wait before next action
            time.sleep(action_interval)
        
        print(f"\n{'='*60}")
        print(f"Execution complete: {len(executed_actions)} actions taken")
        print(f"{'='*60}\n")
        
        return executed_actions
    
    def _execute_action(self, action_plan: Dict) -> bool:
        """Execute a single action"""
        action = action_plan['action']
        params = action_plan['parameters']
        
        try:
            if action == 'stand_up':
                self.robot.stand_up()
                time.sleep(3)
                
            elif action == 'stand_down':
                self.robot.stand_down()
                time.sleep(3)
                
            elif action == 'walk_forward':
                speed = params.get('speed', 0.2)
                duration = params.get('duration', 2.0)
                self.robot.walk_forward(speed, duration)
                
            elif action == 'walk_backward':
                speed = params.get('speed', 0.2)
                duration = params.get('duration', 2.0)
                self.robot.walk_backward(speed, duration)
                
            elif action == 'turn_left':
                speed = params.get('speed', 0.3)
                duration = params.get('duration', 2.0)
                self.robot.turn_left(speed, duration)
                
            elif action == 'turn_right':
                speed = params.get('speed', 0.3)
                duration = params.get('duration', 2.0)
                self.robot.turn_right(speed, duration)
                
            elif action == 'strafe_left':
                speed = params.get('speed', 0.15)
                duration = params.get('duration', 2.0)
                self.robot.strafe_left(speed, duration)
                
            elif action == 'strafe_right':
                speed = params.get('speed', 0.15)
                duration = params.get('duration', 2.0)
                self.robot.strafe_right(speed, duration)
                
            elif action == 'stop':
                self.robot.stop_move()
                
            elif action == 'track_object':
                target = params.get('target', 'person')
                duration = params.get('duration', 5.0)
                # Implement object tracking
                print(f"Tracking {target} for {duration}s")
                
            elif action == 'look_around':
                # Rotate 360 degrees slowly
                self.robot.turn_left(0.3, 4.0)
                
            elif action == 'approach_object':
                self.robot.walk_forward(0.15, params.get('duration', 3.0))
                
            elif action == 'avoid_obstacle':
                self.robot.walk_backward(0.2, 1.5)
                
            elif action == 'none':
                print("No action taken")
                
            else:
                print(f"⚠️  Unknown action: {action}")
                return False
            
            print(f"✓ Executed: {action}")
            return True
            
        except Exception as e:
            print(f"✗ Execution failed: {e}")
            return False


# Example usage functions
def example_with_robot():
    """Example using Gemini with real robot"""
    from g1_high_level_control import G1HighLevelController
    from g1_vision_detection import G1VisionDetector
    
    # Get API key
    api_key = input("Enter your Google API key: ").strip()
    if not api_key:
        print("❌ API key required")
        return
    
    # Setup components
    print("\nSetting up robot...")
    robot = G1HighLevelController()
    robot.connect()
    robot.stand_up()
    
    print("\nSetting up vision...")
    vision = G1VisionDetector(model_type='cascade')
    vision.connect_camera(0)
    vision.start_capture()
    time.sleep(1)
    
    # Setup VLM with Gemini
    print("\nInitializing Gemini VLM...")
    vlm = VLMActionPlanner(api_key=api_key)
    
    # Create VLM controller
    controller = G1VLMController(robot, vision, vlm)
    
    # Execute instruction
    instruction = input("\nEnter instruction (e.g., 'explore the room'): ").strip()
    if not instruction:
        instruction = "Look around and identify objects"
    
    controller.execute_instruction(
        instruction=instruction,
        max_actions=5
    )
    
    # Cleanup
    vision.release()
    robot.stand_down()
    robot.shutdown()


def example_test_only():
    """Example testing Gemini without robot"""
    from g1_vision_detection import G1VisionDetector
    
    # Get API key
    api_key = input("Enter your Google API key: ").strip()
    if not api_key:
        print("❌ API key required. Get one from: https://makersuite.google.com/app/apikey")
        return
    
    # Setup vision only (no robot for testing)
    print("\nSetting up camera...")
    vision = G1VisionDetector(model_type='cascade')
    vision.connect_camera(0)
    vision.start_capture()
    time.sleep(1)
    
    # Setup VLM with Gemini
    print("\nInitializing Gemini VLM...")
    vlm = VLMActionPlanner(api_key=api_key)
    
    # Get a frame and test
    with vision.frame_lock:
        frame = vision.frame.copy()
    
    print("\nQuerying Gemini VLM...")
    action_plan = vlm.decide_action(
        image=frame,
        instruction="Look for people and describe what you see"
    )
    
    print(f"\n{'='*60}")
    print("Gemini VLM Response:")
    print(f"{'='*60}")
    print(f"Action: {action_plan['action']}")
    print(f"Parameters: {json.dumps(action_plan['parameters'], indent=2)}")
    print(f"Reasoning: {action_plan['reasoning']}")
    print(f"Confidence: {action_plan['confidence']:.2%}")
    
    vision.release()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("UNITREE G1 - GEMINI VLM CONTROL")
    print("="*60)
    print("\nThis uses Google Gemini 2.5 Flash for vision + language control")
    print("Get API key from: https://makersuite.google.com/app/apikey")
    
    print("\nSelect mode:")
    print("1. Test with camera only (no robot)")
    print("2. Full robot control with VLM")
    print("3. Show example prompt")
    
    choice = input("\nChoice [1]: ").strip() or "1"
    
    if choice == "1":
        example_test_only()
    elif choice == "2":
        print("\n⚠️  Make sure robot is connected and ready")
        example_with_robot()
    else:
        # Just show how prompt is built
        print("\n" + "="*60)
        print("Example Prompt Structure:")
        print("="*60)
        vlm = VLMActionPlanner(api_key='dummy-key-for-demo')
        prompt = vlm._build_action_prompt(
            instruction="Find and approach the red ball",
            context="Indoor environment, multiple objects visible",
            previous_actions=['stand_up', 'look_around']
        )
        print(prompt)
