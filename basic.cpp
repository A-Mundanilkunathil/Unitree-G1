#include <iostream>
#include "unitree_sdk2/low_level.h"  // adjust include path to your SDK

int main(int argc, char** argv) {
    std::string interface = "enp2s0";  // replace with your network interface
    unitree_sdk2::LowLevelClient client(interface);

    // connect
    if (!client.connect()) {
        std::cerr << "Failed to connect to robot low-level interface\n";
        return -1;
    }

    // Prepare a low-level command structure
    unitree_sdk2::MotorCommand cmd{};
    cmd.mode = unitree_sdk2::MotorMode::PMSM;  // for example
    cmd.q    = 0.0f;     // desired angle in rad
    cmd.dq   = 0.0f;     // desired velocity in rad/s
    cmd.tau  = 0.0f;     // desired torque in N·m
    cmd.Kp   = 10.0f;    // position stiffness
    cmd.Kd   = 1.0f;     // velocity stiffness

    // Example: set one joint
    client.setMotorCommand(joint_index = 0, cmd);

    // send the command
    if (!client.sendCommand()) {
        std::cerr << "Failed to send low-level motor command\n";
    }

    client.disconnect();
    return 0;
}