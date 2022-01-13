import numpy as np
import pybullet as p
import tform as tf
import scipy.linalg as la
import control
import swing_trajectory as st
from scipy import linalg as sci_la

class PreviewControl:
    def __init__(self, dt=1./240., Tsup_time=0.5, Tdl_time=0.1, CoMheight=0.45, g=9.8, previewStepNum=240, stride=0.1, initialTargetZMP=np.array([0.,0.]), initialFootPrint=np.array([[[0.,0.065],[0.,-0.065]]]), R=np.matrix([1.]), Q=np.matrix([[7000,0,0,0],
                                                                                                                                                                                                                                                                [0,1,0,0],
                                                                                                                                                                                                                                                                [0,0,1,0],
                                                                                                                                                                                                                                                                [0,0,0,1]])):
        self._RIGHT_LEG = 1
        self._LEFT_LEG = 0
        self.dt = dt
        self.previewStepNum = previewStepNum
        self.A = np.matrix([[1, dt, (dt**2)/2],
                    [0, 1, dt],
                    [0, 0, 1]])
        self.B = np.matrix([(dt**3)/6, (dt**2)/2, dt]).T
        self.C = np.matrix([1, 0, -CoMheight/g])
        self.CoMheight = CoMheight

        self.G = np.vstack((-self.C*self.B, self.B))
        self.Gr= np.matrix([1., 0., 0., 0.]).T
        #state vector
        self.x = np.matrix(np.zeros(3)).T
        self.y = np.matrix(np.zeros(3)).T

        self.footPrints = np.array([[[0.,0.065],[0.,-0.065]],
                                        [[0.,0.065],[0.,-0.065]],
                                        [[0.,0.065],[0.,-0.065]]])
        self.Tsup = int(Tsup_time/dt)
        self.Tdl = int(Tdl_time/dt)

        self.px_ref = np.full((self.Tsup+self.Tdl)*3,initialTargetZMP[0])
        self.py_ref = np.full((self.Tsup+self.Tdl)*3,initialTargetZMP[1])
        self.px = np.array([0.0]) #zmp
        self.py = np.array([0.0])

        self.phi = np.hstack( (np.matrix([1,0,0,0]).T,  np.vstack((-self.C*self.A, self.A))  ) )
        P, _, _ = control.dare(self.phi,self.G,Q,R)
        zai = (np.eye(4) - self.G * la.inv(R + self.G.T*P*self.G) * self.G.T * P )*self.phi
        self.Fr=np.array([])
        for j in range(1,previewStepNum+1):
            self.Fr= np.append(self.Fr, -la.inv(R + self.G.T*P*self.G)*self.G.T*((zai.T)**(j-1))*P*self.Gr)
        
        self.F=-la.inv(R + self.G.T*P*self.G)*self.G.T*P*self.phi

        self.px_ref_log = self.px_ref[:(self.Tsup+self.Tdl)*2]
        self.py_ref_log = self.py_ref[:(self.Tsup+self.Tdl)*2]

        self.xdu = 0
        self.ydu = 0

        self.xu = 0
        self.yu = 0

        self.dx=np.matrix(np.zeros(3)).T
        self.dy=np.matrix(np.zeros(3)).T


        self.swingLeg = self._RIGHT_LEG
        self.supportLeg = self._LEFT_LEG

        self.targetZMPold = np.array([initialTargetZMP])

        self.currentFootStep = 0
        
        self.kp = 0
        self.kd = 0
        self.Beta_0_prev = [0.00005]*6
        self.Beta_1_prev = [0.00005]*6
        self.c = [0.,0.,0.,0.,0.,0.]
        self.u = [0.,0.,0.,0.,0.,0.]
        self.du = [0.,0.,0.,0.,0.,0.]
        self.alpha = [10.,10.,10.,10.,10.,10.]

    def Mbar(self):
        Massmatrix =  [[0.0004207,0.0,0.0,0.0,0.0,0.0],
                       [0.0,0.0422,0.0,0.0,0.0,0.0],
                       [0.0,0.0,0.0423,0.0,0.0,0.0],
                       [0.0,0.0,0.0,0.0095,0.0,0.0],
                       [0.0,0.0,0.0,0.0,0.00023,0.0],
                       [0.0,0.0,0.0,0.0,0.0,0.000092]]                    
                
        return Massmatrix

    def footPrintAndCOMtrajectoryGenerator(self, inputTargetZMP,inputFootPrint):
        currentFootStep = 0

        self.footPrints = self.footOneStep(self.footPrints,inputFootPrint, self.supportLeg)

        input_px_ref, input_py_ref = self.targetZMPgenerator(inputTargetZMP, self.targetZMPold[-1], self.Tsup,self.Tdl,self.swingLeg)


        self.px_ref = self.fifo(self.px_ref, input_px_ref, len(input_px_ref))
        self.py_ref = self.fifo(self.py_ref, input_py_ref, len(input_py_ref))

        self.px_ref_log = np.append(self.px_ref_log, input_px_ref)
        self.py_ref_log = np.append(self.py_ref_log, input_py_ref)
        
        CoMTrajectory = np.empty((0,3), float)
        startRobotVelocity = np.array([self.x[1],self.y[1]])
        for k in range(len(input_px_ref)):
            dpx_ref = self.px_ref[k+1] - self.px_ref[k]
            dpy_ref = self.py_ref[k+1] - self.py_ref[k]

            xe = self.px_ref[k] - self.C * self.x
            ye = self.py_ref[k] - self.C * self.y

            X=self.phi * np.vstack((xe, self.dx)) + self.G*self.xdu + self.Gr*dpx_ref
            Y=self.phi * np.vstack((ye, self.dy)) + self.G*self.ydu + self.Gr*dpy_ref

            xsum=ysum=0
            for j in range(1,self.previewStepNum+1):
                xsum +=self.Fr[j-1]*(self.px_ref[k+j]-self.px_ref[k+j-1])
                ysum +=self.Fr[j-1]*(self.py_ref[k+j]-self.py_ref[k+j-1])
        
            self.xdu=self.F*X+xsum
            self.ydu=self.F*Y+ysum
            
            self.xu+=self.xdu
            self.yu+=self.ydu
        
            old_x=self.x
            old_y=self.y

            self.x=self.A*self.x+self.B*self.xu
            self.y=self.A*self.y+self.B*self.yu

            self.dx=self.x-old_x
            self.dy=self.y-old_y

            CoMTrajectory = np.vstack((CoMTrajectory, [self.x[0,0], self.y[0,0], self.CoMheight]))

            self.px = np.append(self.px, self.C*self.x)
            self.py = np.append(self.py, self.C*self.y) 

        robotEndVelocity = np.array([self.x[1],self.y[1],0.])

        leftTrj,rightTrj = self.footTrajectoryGenerator(np.hstack((self.footPrints[currentFootStep,self.swingLeg], 0.)),
                                                        np.hstack((self.footPrints[currentFootStep+1,self.swingLeg], 0.)),
                                                        np.array([0.,0.,0.]),
                                                        np.array([0.,0.,0.]),
                                                        np.hstack((self.footPrints[currentFootStep,self.supportLeg],0.)),
                                                        self.swingLeg)

        
        self.swingLeg, self.supportLeg = self.changeSupportLeg(self.swingLeg, self.supportLeg)
        self.targetZMPold = np.vstack((self.targetZMPold, inputTargetZMP))
        

        return CoMTrajectory, leftTrj, rightTrj

    def targetZMPgenerator(self,targetZMP,targetZMPold, Tsup, Tdl,swingLeg):
        tdl_t = np.arange(0,Tdl)

        fr = 0.1*9.8*((tdl_t)/Tsup) 
        fl = 0.1*9.8*(1-((tdl_t)/Tsup))
        
        if swingLeg is self._RIGHT_LEG:
            x_r = targetZMP[0]
            x_l = targetZMPold[0]
            y_r = targetZMP[1]
            y_l = targetZMPold[1]
            
        elif swingLeg is self._LEFT_LEG:
            x_l = targetZMP[0]
            x_r = targetZMPold[0]
            y_l = targetZMP[1]
            y_r = targetZMPold[1]

        xzmp = (x_r*fr + x_l*fl)/(fr + fl)
        yzmp = (y_r*fr + y_l*fl)/(fr + fl) 

        px_ref = np.hstack(( xzmp, np.full(Tsup, targetZMP[0])  ))
        py_ref = np.hstack(( yzmp, np.full(Tsup, targetZMP[1])  ))

        return px_ref, py_ref

    def footTrajectoryGenerator(self,swingStartPointV,swingEndPointV, startRobotVelocityV_xy,endRobotVelocityV,supportPointV,swingLeg,zheight=0.04):
        supportTrajectory = np.vstack((np.full(self.Tdl+self.Tsup,supportPointV[0]),
                                    np.full(self.Tdl+self.Tsup,supportPointV[1]),
                                    np.full(self.Tdl+self.Tsup,supportPointV[2]))).T

        swingTrajectoryForTdl = np.vstack((np.full(self.Tdl,swingStartPointV[0]),
                                            np.full(self.Tdl,swingStartPointV[1]),
                                            np.full(self.Tdl,swingStartPointV[2]))).T

        if np.array_equal(swingStartPointV, swingEndPointV):
            swingTrajectoryForTsup = np.vstack((np.full(self.Tsup,swingEndPointV[0]),
                                            np.full(self.Tsup,swingEndPointV[1]),
                                            np.full(self.Tsup,swingEndPointV[2]))).T
        
        else:
            swingTrajectoryForTsup = st.swingTrajectoryGenerator(swingStartPointV, swingEndPointV, -startRobotVelocityV_xy, -endRobotVelocityV ,zheight, 0.,self.Tsup*self.dt,self.dt) 

        if swingLeg is self._RIGHT_LEG:
            trjR = np.vstack((swingTrajectoryForTdl,swingTrajectoryForTsup))
            trjL = supportTrajectory
            
        elif swingLeg is self._LEFT_LEG:
            trjL = np.vstack((swingTrajectoryForTdl,swingTrajectoryForTsup))
            trjR = supportTrajectory


        return trjL, trjR

    def fifo(self, p, in_p, range, vstack=False):
        if vstack:
            return np.vstack(( np.delete(p, range, 0), in_p ))

        else:
            return np.append( np.delete(p, slice(range), None), in_p )

    def footOneStep(self,footPrints,supportPoint,supportLeg):
        step = len(footPrints)
        if supportLeg is self._LEFT_LEG:
            newFootPrint = np.vstack((footPrints, [np.vstack((supportPoint,footPrints[-1,1]))] ))

        elif supportLeg is self._RIGHT_LEG:
            newFootPrint = np.vstack((footPrints, [np.vstack((footPrints[-1,0], supportPoint))] ))

        return np.delete(newFootPrint, 0, 0)
    

    def changeSupportLeg(self, swingLeg, supportLeg):
        return supportLeg, swingLeg

    def RMSerror(self,qref,q,leg):

        return np.sqrt(((qref-q)**2).mean())

    def errord(self,vref,v,leg):
        return vref-v

    

    def adaptiveRobustSolve1(self,p_ref,p,v,leg,positionGain=90,velocityGain=60,omega = 2,E_bar = 0.7,
                                        Theta_dot = [0.0,0.0,0.0] , Gamma_dot = [0.0,0.0,0.0]):
                    
        Mbar = [[0.042,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.042,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.042,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.042,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.042,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.042]] 
        delta_t = 1/240.                  
        ed =  [0]*len(p_ref)      
        vref = [0]*len(p_ref)
        aref = [0]*len(p_ref)
        torque = [0.,0.,0.,0.,0.,0.]
        aprev = [0]*len(p_ref)
        
        self.kp = positionGain
        self.kd = velocityGain
        self.Gamma = [1.5e-4, 1.5e-4, 1.5e-4]
        self.THETA = [1.5e-4, 5e-5, 3e-5]
        self.THETA_ = [1.5e-4, 5e-5, 3e-5]
        #print(self.kp,self.kd)

        for i in range(len(p_ref)):
            if i == 0:
                vref[i] = 0
            else:
                vref[i] = (p_ref[i]-p_ref[i-1])*(1./240.)

        for i in range(len(p_ref)):
            if i == 0:
                aref[i] = 0
            else:
                aref[i] = (vref[i]-vref[i-1])*(1./240.)
                aprev[i] = aref[i-1]


            e = (p-p_ref)
            ed[i] = (v[i]-vref[i])

            E = np.array([e[i],ed[i]]).T
            A = np.array([[0,1],[-self.kp,-self.kd]])
            Q = -np.ones((2,2))
            P = sci_la.solve_continuous_lyapunov(A, Q)
            B = np.array([0,1])
            s= B @ P @ E

            # wp, vp = nla.eig(P)
            # wq, vq = nla.eig(Q_)
            # zeta = (np.min(wq))/np.max(wp)
            
            # kappa = 0.9*zeta
            # m_lower = 8
            # m_upper = 11
            # mu = m_upper/m_lower
            # art = np.log(mu/kappa) 
            

            Y = np.array([1 , np.linalg.norm(E) , np.square(np.linalg.norm(E))])

            

            Theta_dot_0 = np.linalg.norm(s) - 1*self.THETA[0]
            Theta_dot_1 = (np.linalg.norm(s)*np.linalg.norm(E)) - (1*self.THETA[1])
            Theta_dot_2 = np.square((np.linalg.norm(s)*np.linalg.norm(E))) - (1*self.THETA[2])
            Theta_dot = np.array([Theta_dot_0 , Theta_dot_1, Theta_dot_2])
            #print(Theta_dot)  

            self.THETA = self.THETA + Theta_dot*delta_t   

            # print("1",self.THETA , self.Gamma,Theta_dot)

            rho = (1/(1-E_bar))*(Y @ (self.THETA+self.Gamma))
            dT =  omega*rho*np.multiply(s,(1/np.sqrt((np.square(np.absolute(s)))+0.1)))
            lambda_sigma = np.array([self.kp,self.kd])
        
        if i ==0:
            torque = [0.,0.,0.,0.,0.,0.]
        else:
            temp = lambda_sigma.T@E
            # temp = temp.T
            
            torque = np.dot(Mbar,(temp - dT + aref[i]))
            # print("1",torque)
        return torque
        # ,beta_0,beta_1,self.c*18


    def adaptiveRobustSolve2(self,p_ref,p,v,leg,positionGain=70,velocityGain=40,omega =1,E_bar = 0.7,
                            Theta_dot = [0.0,0.0,0.0] , Gamma_dot = [0.0,0.0,0.0]):
        
        Mbar = [[0.042,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.042,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.042,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.042,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.042,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.042]]
        delta_t = 1./240.         
        ed =  [0]*len(p_ref)      
        vref = [0]*len(p_ref)
        aref = [0]*len(p_ref)
        torque = [0.,0.,0.,0.,0.,0.]
        aprev = [0]*len(p_ref)
        
        self.kp = positionGain
        self.kd = velocityGain
        self.Gamma = [1.5e-4, 1.5e-4, 1.5e-4]
        self.THETA = [1.5e-4, 5e-5, 3e-5]


        for i in range(len(p_ref)):
            if i == 0:
                vref[i] = 0
            else:
                vref[i] = (p_ref[i]-p_ref[i-1])*delta_t

        for i in range(len(p_ref)):
            if i == 0:
                aref[i] = 0
            else:
                aref[i] = (vref[i]-vref[i-1])*delta_t
                aprev[i] = aref[i-1]

            e = (p-p_ref)
            ed[i] = (v[i]-vref[i])
 
            E = np.array([e[i],ed[i]]).T
            A = np.array([[0,1],[-self.kp,-self.kd]])
            Q = -np.ones((2,2))
            P = sci_la.solve_continuous_lyapunov(A, Q)
            B = np.array([0,1])
            s= B @ P @ E            # wp, vp = nla.eig(P)
            # wq, vq = nla.eig(Q_)
            # zeta = (np.min(wq))/np.max(wp)
            
            # kappa = 0.9*zeta
            # m_lower = 8
            # m_upper = 11
            # mu = m_upper/m_lower
            # art = np.log(mu/kappa) 
            #print(art)
            # print(np.shape(E))
            E_norm = np.linalg.norm(E)
            Y = np.array([1 , E_norm , np.square(E_norm)])

    
            beta = 10
            v_bar = 1e-1
            vp_bar = 1e-1

            G_dot_0 = -np.multiply((beta + np.multiply(v_bar,np.power(self.THETA[0],4))),self.Gamma[0]) + np.multiply(beta,vp_bar)
            G_dot_1 = -np.multiply((beta + np.multiply(v_bar,np.power(self.THETA[1],4))),self.Gamma[1]) + np.multiply(beta,vp_bar)
            G_dot_2 = -np.multiply((beta + np.multiply(v_bar,np.power(self.THETA[2],4))),self.Gamma[2]) + np.multiply(beta,vp_bar)
            Gamma_dot = np.array([G_dot_0 , G_dot_1, G_dot_2])


            self.Gamma = self.Gamma + Gamma_dot*delta_t    

            # print("2",self.Gamma)

            rho = (1/(1-E_bar))*(Y @ (self.THETA+self.Gamma))
            dT =  omega*rho*np.multiply(s,(1/np.sqrt((np.square(np.absolute(s)))+0.1)))
            lambda_sigma = np.array([self.kp,self.kd])
            # print(dT)
        
        if i == 0:
            torque = [0.,0.,0.,0.,0.,0.]
        else:
    
            temp = lambda_sigma.T@E
            # temp = temp.T
            
            torque = np.dot(Mbar,(temp - dT + aref[i]))
            # torque = torque.diagonal()
            # print("2",torque)
        return torque
    


 