#!/usr/bin/env python3
"""
G1 Natural Language Control with LLM
Takes natural language input, uses LLM to interpret it, and executes precise movements.
"""

import time
import sys
import os
import json
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
import math

# Try to import python-dotenv for loading .env files
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️  python-dotenv not available, install with: pip install python-dotenv")

# Try to import OpenAI (also works for Groq via compatible API)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI library not available, install with: pip install openai")

# Try to import Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Gemini not available, install with: pip install google-generativeai")


class MovementController:
    """Handles all G1 robot movements"""
    
    def __init__(self):
        self.loco_client = None
        self.arm_client = None
        self.walk_speed = 0.3  # meters per second
        self.turn_speed = 0.3  # radians per second

    def initialize_clients(self):
        """Initialize locomotion and arm clients"""
        try:
            self.loco_client = LocoClient()
            self.loco_client.SetTimeout(10.0)
            self.loco_client.Init()
            
            self.arm_client = G1ArmActionClient()
            self.arm_client.SetTimeout(10.0)
            self.arm_client.Init()
            
            print("✓ Movement controllers initialized")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize: {e}")
            return False

    def walk_distance(self, distance, direction="forward"):
        """Walk a specific distance"""
        try:
            if direction == "forward":
                vx, vy, vyaw = self.walk_speed, 0, 0
            elif direction == "backward":
                vx, vy, vyaw = -self.walk_speed, 0, 0
            elif direction == "left":
                vx, vy, vyaw = 0, self.walk_speed, 0
            elif direction == "right":
                vx, vy, vyaw = 0, -self.walk_speed, 0
            
            duration = abs(distance) / self.walk_speed
            
            print(f"🚶 Walking {direction} {distance}m...")
            self.loco_client.Move(vx, vy, vyaw)
            time.sleep(duration)
            print("✓ Done")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def turn_angle(self, angle_degrees, direction="left"):
        """Turn by specific angle"""
        try:
            angle_rad = math.radians(abs(angle_degrees))
            vyaw = self.turn_speed if direction == "left" else -self.turn_speed
            duration = angle_rad / self.turn_speed
            
            print(f"🔄 Turning {direction} {angle_degrees}°...")
            self.loco_client.Move(0, 0, vyaw)
            time.sleep(duration)
            print("✓ Done")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def perform_gesture(self, gesture):
        """Perform arm gestures"""
        try:
            print(f"👋 {gesture}...")
            
            if gesture == "wave":
                self.arm_client.ExecuteAction(action_map.get("high wave"))
                time.sleep(3)
            elif gesture == "clap":
                self.arm_client.ExecuteAction(action_map.get("clap"))
                time.sleep(2)
            elif gesture == "thumbs up":
                self.arm_client.ExecuteAction(action_map.get("right hand up"))
                time.sleep(2)
            elif gesture == "peace":
                self.arm_client.ExecuteAction(action_map.get("face wave"))
                time.sleep(2)
            elif gesture == "flex":
                self.arm_client.ExecuteAction(action_map.get("hands up"))
                time.sleep(2)
            elif gesture == "stretch":
                self.arm_client.ExecuteAction(action_map.get("x-ray"))
                time.sleep(3)
            
            self.arm_client.ExecuteAction(action_map.get("release arm"))
            time.sleep(1)
            print("✓ Done")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def posture(self, posture_type):
        """Change posture"""
        try:
            print(f"🧍 {posture_type}...")
            
            if posture_type == "squat":
                self.loco_client.StandUp2Squat()
                time.sleep(2)
            elif posture_type == "stand":
                self.loco_client.Squat2StandUp()
                time.sleep(2)
            elif posture_type == "bow":
                self.loco_client.StandUp2Squat()
                time.sleep(2)
                self.loco_client.Squat2StandUp()
                time.sleep(2)
            
            print("✓ Done")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False


class LLMInterpreter:
    """Uses LLM to interpret natural language and map to movement commands"""
    
    def __init__(self, use_groq=True):
        self.use_groq = use_groq
        self.use_openai = False
        self.client = None
        self.model_name = "llama-3.3-70b-versatile"  # Production model: 70B, fast, reliable
        
        # Available commands for the robot
        self.available_commands = {
            "walk_forward": "Walk forward X meters (1-5)",
            "walk_backward": "Walk backward X meters (1-5)",
            "walk_left": "Walk left X meters (1-2)",
            "walk_right": "Walk right X meters (1-2)",
            "turn_left": "Turn left X degrees (45, 90, 180)",
            "turn_right": "Turn right X degrees (45, 90, 180)",
            "wave": "Wave hand gesture",
            "clap": "Clap hands",
            "thumbs_up": "Give thumbs up",
            "peace": "Peace sign gesture",
            "flex": "Flex muscles",
            "stretch": "Stretch arms",
            "bow": "Bow greeting",
            "squat": "Squat down",
            "stand": "Stand up tall",
            "stop": "Stop all movement"
        }
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        # Try Groq first (ultra-fast inference)
        if self.use_groq and OPENAI_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                try:
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.groq.com/openai/v1"
                    )
                    self.model_name = "llama-3.3-70b-versatile"
                    # Test connection
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=5
                    )
                    print("✓ Groq LLM initialized (llama-3.3-70b-versatile)")
                    print("  Production model: 70B params, 280 tps, 131K context")
                    return
                except Exception as e:
                    print(f"⚠️  Groq failed: {e}")
        
        # Fallback to OpenAI
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.client = OpenAI(api_key=api_key)
                    self.model_name = "gpt-3.5-turbo"
                    self.use_openai = True
                    # Test connection
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=5
                    )
                    print("✓ OpenAI LLM initialized")
                    return
                except Exception as e:
                    print(f"⚠️  OpenAI failed: {e}")
        
        if GEMINI_AVAILABLE:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.client = genai.GenerativeModel('gemini-pro')
                    self.use_openai = False
                    print("✓ Gemini LLM initialized")
                    return
                except Exception as e:
                    print(f"⚠️  Gemini failed: {e}")
        
        print("⚠️  No LLM available - using pattern matching only")
        self.client = None
    
    def interpret_command(self, user_input: str) -> dict:
        """
        Interpret natural language and return structured command
        Returns: {"action": str, "params": dict}
        """
        if self.client:
            return self._llm_interpret(user_input)
        else:
            return self._pattern_match(user_input)
    
    def _llm_interpret(self, user_input: str) -> dict:
        """Use LLM to interpret command"""
        prompt = f"""You are controlling a humanoid robot. Convert the user's command to a structured robot action.

User command: "{user_input}"

Available robot actions:
{json.dumps(self.available_commands, indent=2)}

Respond with ONLY a JSON object in this format:
{{
  "action": "action_name",
  "params": {{"param1": value1, "param2": value2}}
}}

Examples:
- "walk 2 meters forward" → {{"action": "walk_forward", "params": {{"distance": 2}}}}
- "turn left 90 degrees" → {{"action": "turn_left", "params": {{"angle": 90}}}}
- "wave hello" → {{"action": "wave", "params": {{}}}}
- "take 3 steps back" → {{"action": "walk_backward", "params": {{"distance": 1}}}}

Respond ONLY with the JSON, no other text."""

        try:
            if self.use_groq or self.use_openai:
                # Both Groq and OpenAI use the same API format
                response = self.client.chat.completions.create(
                    model=self.model_name,  # llama-3.3-70b-versatile or gpt-3.5-turbo
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=200
                )
                result = response.choices[0].message.content
            else:
                # Gemini fallback
                response = self.client.generate_content(prompt)
                result = response.text
            
            # Extract JSON from response
            start = result.find('{')
            end = result.rfind('}') + 1
            if start != -1 and end > start:
                json_str = result[start:end]
                return json.loads(json_str)
            
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
        
        # Fallback to pattern matching
        return self._pattern_match(user_input)
    
    def _pattern_match(self, user_input: str) -> dict:
        """Simple pattern matching fallback"""
        cmd = user_input.lower()
        
        # Extract numbers
        import re
        numbers = re.findall(r'\d+', cmd)
        distance = int(numbers[0]) if numbers else 1
        angle = int(numbers[0]) if numbers else 90
        
        # Movement patterns
        if 'forward' in cmd or 'ahead' in cmd:
            return {"action": "walk_forward", "params": {"distance": min(distance, 5)}}
        elif 'back' in cmd or 'backward' in cmd:
            return {"action": "walk_backward", "params": {"distance": min(distance, 5)}}
        elif 'turn left' in cmd or 'rotate left' in cmd:
            return {"action": "turn_left", "params": {"angle": angle}}
        elif 'turn right' in cmd or 'rotate right' in cmd:
            return {"action": "turn_right", "params": {"angle": angle}}
        elif 'left' in cmd:
            return {"action": "walk_left", "params": {"distance": min(distance, 2)}}
        elif 'right' in cmd:
            return {"action": "walk_right", "params": {"distance": min(distance, 2)}}
        
        # Gesture patterns
        elif 'wave' in cmd or 'hello' in cmd or 'hi ' in cmd:
            return {"action": "wave", "params": {}}
        elif 'clap' in cmd:
            return {"action": "clap", "params": {}}
        elif 'thumb' in cmd:
            return {"action": "thumbs_up", "params": {}}
        elif 'peace' in cmd:
            return {"action": "peace", "params": {}}
        elif 'flex' in cmd or 'muscle' in cmd:
            return {"action": "flex", "params": {}}
        elif 'stretch' in cmd:
            return {"action": "stretch", "params": {}}
        elif 'bow' in cmd:
            return {"action": "bow", "params": {}}
        
        # Posture patterns
        elif 'squat' in cmd or 'crouch' in cmd:
            return {"action": "squat", "params": {}}
        elif 'stand' in cmd or 'up' in cmd:
            return {"action": "stand", "params": {}}
        elif 'stop' in cmd:
            return {"action": "stop", "params": {}}
        
        return {"action": "unknown", "params": {}}


def main():
    # Load environment variables from .env file
    if DOTENV_AVAILABLE:
        load_dotenv()
        print("✓ Loaded environment variables from .env")
    else:
        print("⚠️  Install python-dotenv to load API keys from .env file")
        print("   Run: pip install python-dotenv")
    
    # Verify at least one API key is available
    if not any([
        os.getenv("GROQ_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GEMINI_API_KEY")
    ]):
        print("\n❌ No API keys found!")
        print("   Please add your API keys to the .env file")
        print("   See .env.example for template\n")
    
    if len(sys.argv) < 2:
        print("\nUsage: python3 g1_nlp_control.py <network_interface>")
        print("Example: python3 g1_nlp_control.py eth0\n")
        sys.exit(1)
    
    network_interface = sys.argv[1]
    
    print("\n" + "="*60)
    print("🤖 G1 Natural Language Control with Groq LLM")
    print("="*60)
    print(f"Network: {network_interface}")
    print(f"Model: llama-3.3-70b-versatile (Production)")
    print("  • Meta Llama 3.3 70B - Latest production model")
    print("  • 280 tokens/sec - Ultra-fast inference")
    print("  • 131K context window - Large context support")
    print("\n⚠️  Robot will execute commands!")
    print("="*60 + "\n")
    
    input("Press Enter to connect...")
    
    # Initialize DDS
    ChannelFactoryInitialize(0, network_interface)
    
    # Initialize movement controller
    controller = MovementController()
    if not controller.initialize_clients():
        print("❌ Failed to initialize robot")
        sys.exit(1)
    
    # Initialize LLM interpreter
    llm = LLMInterpreter(use_groq=True)
    
    print("\n" + "="*60)
    print("Ready! Type natural language commands")
    print("="*60)
    print("\nExamples:")
    print("  'walk forward 2 meters'")
    print("  'turn left 90 degrees'")
    print("  'wave hello'")
    print("  'take 3 steps back'")
    print("  'rotate right 180'")
    print("\nType 'quit' to exit")
    print("="*60 + "\n")
    
    command_count = 0
    
    while True:
        try:
            user_input = input("🎤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Shutting down...")
                break
            
            command_count += 1
            print(f"\n[Command {command_count}]")
            
            # Interpret command
            command = llm.interpret_command(user_input)
            action = command.get("action")
            params = command.get("params", {})
            
            print(f"💭 Interpreted: {action} {params}")
            
            # Execute command
            if action == "walk_forward":
                controller.walk_distance(params.get("distance", 1), "forward")
            elif action == "walk_backward":
                controller.walk_distance(params.get("distance", 1), "backward")
            elif action == "walk_left":
                controller.walk_distance(params.get("distance", 1), "left")
            elif action == "walk_right":
                controller.walk_distance(params.get("distance", 1), "right")
            elif action == "turn_left":
                controller.turn_angle(params.get("angle", 90), "left")
            elif action == "turn_right":
                controller.turn_angle(params.get("angle", 90), "right")
            elif action in ["wave", "clap", "thumbs_up", "peace", "flex", "stretch"]:
                controller.perform_gesture(action)
            elif action in ["bow", "squat", "stand"]:
                controller.posture(action)
            elif action == "stop":
                controller.loco_client.Move(0, 0, 0)
                print("✓ Stopped")
            elif action == "unknown":
                print("❓ Didn't understand that command")
            else:
                print(f"❓ Unknown action: {action}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
    
    # Cleanup
    print("\n🔄 Shutting down...")
    try:
        controller.loco_client.Move(0, 0, 0)
        print("✓ Stopped robot")
    except:
        pass
    
    print(f"📊 Total commands: {command_count}")
    print("👋 Goodbye!\n")


if __name__ == "__main__":
    main()
