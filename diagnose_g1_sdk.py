"""
G1 SDK Diagnostic - Find the Correct API
Run this on the robot to discover G1-specific modules
"""

import sys

print("="*60)
print("UNITREE G1 SDK DIAGNOSTIC")
print("="*60)

# Check SDK installation
print("\n1. Checking SDK installation...")
try:
    import unitree_sdk2py
    print(f"✓ unitree_sdk2py version: {getattr(unitree_sdk2py, '__version__', 'unknown')}")
    print(f"✓ Location: {unitree_sdk2py.__file__}")
except ImportError as e:
    print(f"✗ SDK not found: {e}")
    sys.exit(1)

# List top-level modules
print("\n2. Top-level modules in SDK:")
modules = [x for x in dir(unitree_sdk2py) if not x.startswith('_')]
for mod in sorted(modules):
    print(f"   - {mod}")

# Check for robot-specific modules
print("\n3. Checking for robot-specific modules:")
robot_types = ['g1', 'h1', 'go2', 'b2', 'humanoid', 'quadruped']
for robot in robot_types:
    try:
        mod = getattr(unitree_sdk2py, robot, None)
        if mod:
            print(f"✓ {robot}: {mod}")
            submods = [x for x in dir(mod) if not x.startswith('_')]
            for submod in submods[:5]:  # Show first 5
                print(f"     └─ {submod}")
        else:
            print(f"✗ {robot}: not found")
    except Exception as e:
        print(f"✗ {robot}: error - {e}")

# Check for sport/control modules
print("\n4. Checking control modules:")
try:
    from unitree_sdk2py.go2.sport import sport_client
    print("✓ go2.sport.sport_client (QUADRUPED - wrong for G1!)")
    print(f"   SportClient methods:")
    client_methods = [x for x in dir(sport_client.SportClient) if not x.startswith('_')]
    for method in sorted(client_methods)[:10]:
        print(f"     - {method}")
except ImportError as e:
    print(f"✗ go2.sport: {e}")

# Check for humanoid-specific modules
print("\n5. Searching for humanoid/G1-specific controllers...")
try:
    import pkgutil
    import importlib
    
    for importer, modname, ispkg in pkgutil.walk_packages(
        path=unitree_sdk2py.__path__,
        prefix=unitree_sdk2py.__name__ + '.',
        onerror=lambda x: None
    ):
        if any(keyword in modname.lower() for keyword in ['g1', 'h1', 'humanoid', 'biped']):
            print(f"✓ Found: {modname}")
            try:
                mod = importlib.import_module(modname)
                items = [x for x in dir(mod) if not x.startswith('_')][:5]
                for item in items:
                    print(f"     └─ {item}")
            except:
                pass
except Exception as e:
    print(f"✗ Search failed: {e}")

# Check IDL messages
print("\n6. Checking available message types...")
try:
    from unitree_sdk2py.idl import unitree_go
    print("✓ IDL messages available:")
    print(f"   {dir(unitree_go.msg)[:5]}")
except ImportError as e:
    print(f"✗ IDL messages: {e}")

# Summary
print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\n📋 Action Items:")
print("1. Look for modules with 'g1', 'h1', or 'humanoid' in the name above")
print("2. If only 'go2' modules exist, contact Unitree for G1-specific SDK")
print("3. Check /opt/unitree/ or robot docs for G1 examples")
print("4. Try Unitree's GitHub: github.com/unitreerobotics")
print("\n⚠️  IMPORTANT: go2.sport.SportClient is for QUADRUPEDS, not humanoids!")
