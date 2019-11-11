import URBasic
import time
import NetFT
import numpy as np

s2tcptran = np.array([[0.4946, -0.8697, 0], [0.8697, 0.4946, 0],
                      [0, 0, 1]])  # sensor to tcp rotation matrix,expressed in sensor frame
G = np.array([0, 0, -11.1724])  # tool gravity,expressed in base frame
p = np.array([[0, -0.0569,  0.0066], [0.0569, 0, 0.0011],
              [- 0.0066, -0.0011, 0]])  # mass center in sensor frame
# sensor initial offset
fd = np.array([-9.2318, 4.4264, -1.2309,  0.0192,    0.5157,      0.4265])
tcp2sensor = np.linalg.inv(gra_comp.s2tcptran)


class ImpedanceController:
    def __init__(self):
        self.robot = URBasic.urScriptExt.UrScriptExt(
            host='192.168.1.50', robotModel=URBasic.robotModel.RobotModel())
        self.sensor = NetFT.Sensor('192.168.1.30')
        self.M = np.diag((1.0, 1.0, 1.0, 0.1, 0.1, 0.1))
        self.B = np.diag((160.0, 160.0, 160.0, 6.0, 6.0, 6.0))
        self.v = [0, 0, 0, 0, 0, 0]
        self.vd = [0, 0, 0, 0, 0, 0]
        self.Tc = 0.008
        self.MNum = np.linalg.inv(self.M+self.B*self.Tc)

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
        return ftcomp

    def run(self, flag):
        count = 1
        init_pose = self.robot.get_actual_tcp_pose()
        self.robot.movel(pose=init_pose, a=1.2, v=1.0)
        z0 = init_pose[2]
        while count <= 6000:
            ft = ft = np.array(self.sensor.tare()) / 1000000.0
            axisangle = self.robot.get_target_tcp_pose()[3:6]
            rotate = URBasic.kinematic.AxisAng2RotaMatri(axisangle)
            ftcomp = gravitycomp(ft, np.linalg.inv(rotate))
            base2sensor = np.dot(rotate, tcp2sensor)
            base2sensor1 = np.c_[base2sensor, zero]
            base2sensor2 = np.c_[zero, base2sensor]
            base2sensortran = np.r_[base2sensor1, base2sensor2]
            ft_base = np.reshape(np.dot(base2sensortran, ftcomp), (6, 1))
            print(ft_base)
            if count <= 1:
                self.vd = np.reshape(self.vd, (6, 1))
            self.v = np.dot(self.MNum * self.Tc, ft_base) + np.dot(self.MNum * self.M, self.vd)
            self.vd = self.v
            count += 1
            for i in range(2):
                if abs(ft_base[i] < 0.5 or abs(ft_base[i]) > 50):
                    self.v[i] = 0
            for i in range(3, 6):
                if abs(ft_base[i] < 0.1 or abs(ft_base[i]) > 20):
                    self.v[i] = 0
            if flag is True:
                self.v[2] = -(20-ft_base[2])*0.0003
                self.v[3:6] = -self.v[3:6]
                if abs(self.robot.get_actual_tcp_pose()[2]-z0) > 0.036:
                    self.robot.stopl()
                    break
            self.robot.speedl(xd=self.v)
        self.robot.stopl()
        self.robot.close()


if __name__ == "__main__":
    controller = ImpedanceController()
    controller.run(flag=False)  # flag=true，执行插孔，否则执行示教
