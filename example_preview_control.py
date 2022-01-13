import numpy as np
from robot import Bipedal
import tform as tf
import time
from walking_generator import PreviewControl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json


if __name__ == "__main__":
    bipedal = Bipedal() 
    zc = 0.45 #Center of Mass height [m]
    stride = 0.1
    CoM_to_body = np.array([0.0, 0.0, 0.0]) #from CoM to body coordinate

    targetRPY = [0.0, 0.0, 0.0]
    pre = PreviewControl(Tsup_time=0.3,Tdl_time=0.1, previewStepNum=190)#preview control
    bipedal.positionInitialize(initializeTime=0.2)
    start = 0.0
    end = 48.0
    t = np.linspace(start,end,num = 4800)
    p = np.linspace(start,49.92,num = 4992)
    q = np.linspace(start,48.01,num = 4801)
    CoMTrajectory = np.empty((0,3), float)
    trjR_log = np.empty((0,3), float)
    trjL_log = np.empty((0,3), float)
    trjJoint = np.empty((0,6),float)
    trjError = np.empty((0,6),float)
    grfl = np.empty((0,1), float)
    grfr = np.empty((0,1), float)
    C = np.empty((0,1),float)
    beta0 = np.empty((0,1),float)
    beta1 = np.empty((0,1),float)
    trjTor = np.empty((0,6),float)
    erRMS = 0.0
    T = 0.0
    erM = np.empty((0,6),float)
    ersq = np.empty((0,6),float)
    RMS = np.empty((0,6),float)
    MAE = np.empty((0,6),float)
    colours = ['red',
            'brown',
            'cyan',
            'pink',
            'purple',
            'orange']
    joint_labels = ['r_yaw_hipJoint',
                'r_roll_hipJoint',
                'r_pitch_hipJoint',
                'knee',
                'r_pitch_ankleJoint',
                'r_roll_ankleJoint']
    walkingCycle = 50
    supPoint = np.array([0.,0.065])


    for w in range(walkingCycle):

        if w%2 == 0:
            switch = 1
        else:
            switch = 0

        comTrj,footTrjL,footTrjR = pre.footPrintAndCOMtrajectoryGenerator(inputTargetZMP=supPoint, inputFootPrint=supPoint) #generate one cycle trajectory

        
        CoMTrajectory = np.vstack((CoMTrajectory, comTrj))
        trjR_log = np.vstack((trjR_log, footTrjR ))
        trjL_log = np.vstack((trjL_log, footTrjL )) 

        com_len = len(comTrj)
        for i in range(com_len):
            targetPositionR = footTrjR[i] - comTrj[i]
            targetPositionL = footTrjL[i] - comTrj[i]

            PosR = bipedal.inverseKinematics(targetPositionR, targetRPY, bipedal.R)
            PosL = bipedal.inverseKinematics(targetPositionL, targetRPY, bipedal.L)
            objPR = bipedal.getJointPositions(bipedal.R)
            objPL = bipedal.getJointPositions(bipedal.L)
            objVR = bipedal.getJointVelocities(bipedal.R)
            objVL = bipedal.getJointVelocities(bipedal.L)
            
            er = ((PosR-objPR)*30)

            if(switch == 1):
                print('1')
                torR = bipedal.adaptiveRobust2(PosR,objPR,objVR,bipedal.R)
                torL = bipedal.adaptiveRobust1(PosL,objPL,objVL,bipedal.L)
                trjTor = np.vstack((trjTor, torR))

            if(switch == 0):
                print('2')

                torR = bipedal.adaptiveRobust1(PosR,objPR,objVR,bipedal.R)
                torL = bipedal.adaptiveRobust2(PosL,objPL,objVL,bipedal.L)
                trjTor = np.vstack((trjTor, torR))

            # torR = bipedal.adaptiveRobust1(PosR,objPR,objVR,bipedal.R)
            # torL = bipedal.adaptiveRobust1(PosL,objPL,objVL,bipedal.L)
            # trjTor = np.vstack((trjTor, torR))
           
            bipedal.setLeftLegJointPositions(PosL,torL)
            bipedal.setRightLegJointPositions(PosR,torR)

            bipedal.oneStep()

        supPoint[0] += stride
        supPoint[1] = -supPoint[1]


    bipedal.disconnect()

