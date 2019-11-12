#!/usr/bin/python3
import URBasic
import time
import math
import NetFT
import numpy as np

s2tcptran = np.array([[0.5179, -0.8959, 0], [0.8959, 0.5179, 0],
                      [0, 0, 1]])  # sensor to tcp rotation matrix,expressed in sensor frame
G = np.array([0, 0, -11.1431])  # tool gravity,expressed in base frame
p = np.array([[0, -0.0545,  0.0034], [0.0545, 0, 0.0023],
              [- 0.0034, -0.0023, 0]])  # mass center in sensor frame
# sensor initial offset
fd = np.array([-0.3819, -1.3764, -8.9912,  -0.0131,    -0.0841,      -0.0328])
tcp2sensor = np.linalg.inv(s2tcptran)
zero = np.zeros((3, 3), dtype=float)


class ImpedanceController:
    def __init__(self):
        self.robot = URBasic.urScriptExt.UrScriptExt(
            host='192.168.1.50', robotModel=URBasic.robotModel.RobotModel())
        print(self.robot.get_actual_tcp_pose()[3:6])
        self.sensor = NetFT.Sensor('192.168.1.30')
        self.M = np.diag((0.008, 0.008, 0.008, 0.008, 0.008, 0.008))
        self.B = np.diag((80, 80, 80, 6, 6, 6))
        self.v = [0, 0, 0, 0, 0, 0]
        self.vd = [0, 0, 0, 0, 0, 0]
        self.Tc = 0.008
        self.MNum = np.linalg.inv(self.M+self.B*self.Tc)

    def AxisAng2RotaMatri(self, angle_vec):
        '''
        Convert an Axis angle to rotation matrix
        AxisAng2Matrix(angle_vec)
        angle_vec need to be a 3D Axis angle
        '''
        theta = math.sqrt(angle_vec[0]**2+angle_vec[1]**2+angle_vec[2]**2)
        if theta == 0.:
            return np.identity(3, dtype=float)

        cs = np.cos(theta)
        si = np.sin(theta)
        e1 = angle_vec[0]/theta
        e2 = angle_vec[1]/theta
        e3 = angle_vec[2]/theta

        R = np.zeros((3, 3))
        R[0, 0] = (1-cs)*e1**2+cs
        R[0, 1] = (1-cs)*e1*e2-e3*si
        R[0, 2] = (1-cs)*e1*e3+e2*si
        R[1, 0] = (1-cs)*e1*e2+e3*si
        R[1, 1] = (1-cs)*e2**2+cs
        R[1, 2] = (1-cs)*e2*e3-e1*si
        R[2, 0] = (1-cs)*e1*e3-e2*si
        R[2, 1] = (1-cs)*e2*e3+e1*si
        R[2, 2] = (1-cs)*e3**2+cs
        return R

    def gravitycomp(self, ft, rotate_i):
        '''
         this is a gravit compensation function to compensate the tool gravity and obtain the precise
        force on the load.
        '''
        # axisangle = self.robot.get_target_tcp_pose()[3:6]
        # rotate = URBasic.kinematic.AxisAng2RotaMatri(axisangle)
        # tcp rotation matrix expressed in tcp frame
        trans = np.dot(s2tcptran, rotate_i)
        # in sensor frame
        Ftrans = np.dot(trans, G)
        Mtrans = np.dot(p, Ftrans)
        fttrans = np.hstack((Ftrans, Mtrans))
        ftcomp = ft-fttrans-fd
        # print('ok')
        return ftcomp

    def run(self, flag):
        count = 1
        init_pose = self.robot.get_actual_tcp_pose()
        self.robot.movel(pose=init_pose, a=1.2, v=1.0)
        z0 = init_pose[2]
        while count <= 6000:
            ft = np.array(self.sensor.tare()) / 1000000.0
            pose = self.robot.get_actual_tcp_pose()
            axisangle = pose[-3:]
            rotate = self.AxisAng2RotaMatri(axisangle)
            ftcomp = self.gravitycomp(ft, np.linalg.inv(rotate))
            base2sensor = np.dot(rotate, tcp2sensor)
            base2sensor1 = np.c_[base2sensor, zero]
            base2sensor2 = np.c_[zero, base2sensor]
            base2sensortran = np.r_[base2sensor1, base2sensor2]
            ft_base = np.reshape(np.dot(base2sensortran, ftcomp), (6, 1))
            # print(ft_base)
            if count <= 1:
                self.vd = np.reshape(self.vd, (6, 1))
            self.v = np.dot(self.MNum * self.Tc, ft_base) + np.dot(self.MNum * self.M, self.vd)
            self.vd = self.v
            count += 1
            # for i in range(3):
            #     if abs(ft_base[i] < 0.5 or abs(ft_base[i]) > 50):
            #         self.v[i] = 0
            # for i in range(3, 6):
            #     if abs(ft_base[i] < 0.1 or abs(ft_base[i]) > 20):
            #         self.v[i] = 0
            for i in range(3):
                if abs(self.v[i]) < 0.05 or abs(self.v[i]) > 1.5:
                    self.v[i] = 0
            for i in range(3, 6):
                if abs(self.v[i]) < 0.01 or abs(self.v[i]) > 1.0:
                    self.v[i] = 0
            if flag is True:
                self.v[2] = -(20-ft_base[2])*0.0003
                self.v[3:6] = -self.v[3:6]
                if abs(self.robot.get_actual_tcp_pose()[2]-z0) > 0.036:
                    self.robot.stopl()
                    break
            v_tmp = self.v.tolist()
            v_cmd = [i for item in v_tmp for i in item]
            print(v_cmd)
            self.robot.speedl(xd=v_cmd, wait=False, a=0.8)
            # time.sleep(0.008)
        self.robot.stopl()
        self.robot.close()

    def comptest(self):
        ft = np.array(self.sensor.tare()) / 1000000.0
        pose = self.robot.get_actual_tcp_pose()
        axisangle = pose[-3:]
        rotate = self.AxisAng2RotaMatri(axisangle)
        ftcomp = self. gravitycomp(ft, np.linalg.inv(rotate))
        print (ftcomp)


if __name__ == "__main__":
    controller = ImpedanceController()
    controller.run(flag=False)  # flag=true,peg-inhole else teach_mode
    # controller.comptest()
