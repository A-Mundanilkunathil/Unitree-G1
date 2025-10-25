from unitree_sdk2 import LowLevelClient, MotorCommand, MotorMode

def main():
    interface = "enp2s0"  # change as appropriate
    client = LowLevelClient(interface)
    client.connect()

    cmd = MotorCommand()
    cmd.mode = MotorMode.PMSM
    cmd.q    = 0.0     # rad
    cmd.dq   = 0.0     # rad/s
    cmd.tau  = 0.0     # N·m
    cmd.Kp   = 10.0
    cmd.Kd   = 1.0

    joint_index = 0
    client.set_motor_command(joint_index, cmd)
    client.send_command()

    client.disconnect()

if __name__ == '__main__':
    main()