import numpy as np


class Controller():
    def __init__(self):
        self.gain = 2.0
        #pass

    def angle_control_commands(self, dist, angle, v, controller='P'):
        # Return the angular velocity in order to control the Duckiebot so that it follows the lane.
        # Parameters:
        #     dist: distance from the center of the lane. Left is negative, right is positive.
        #     angle: angle from the lane direction, in rad. Left is negative, right is positive.
        # Outputs:
        #     omega: angular velocity, in rad/sec. Right is negative, left is positive.
        k_P=35
        k_D=500
        k_I=3
        
        if controller=='P':
            ####### P
            omega = k_P**2*v*dist+k_P*angle  
            #######
        elif controller=='PD':
            ####### PD
            if not hasattr(self, 'temp_dist'):
                # if first itteration derivative = 0
                self.temp_dist = dist        
                self.temp_angle = angle

            omega_P = k_P**2*v*dist+k_P*angle
            omega_D = k_D*(dist-self.temp_dist)
            omega= omega_P+omega_D

            self.temp_dist=dist
            self.temp_angle=angle
            #######

        ####### PDI
    #     if i==0:
    #         temp_dist = dist        
    #         temp_angle = angle

    #     idx=max(0,i-9)

    #     omega_P = k_P**2*0.5*dist+k_P*angle
    #     omega_I = k_I*np.nanmean(list_dist[idx:i+1])
    #     omega_D = k_D*(temp_dist-dist)

    #     omega= omega_P+omega_D+omega_I

    #     temp_dist=dist
    #     temp_angle=angle
        #######

        return  omega

    def pure_pursuit(self, env, pos, angle, follow_dist=0.25):
        # Return the angular velocity in order to control the Duckiebot using a pure pursuit algorithm.
        # Parameters:
        #     env: Duckietown simulator
        #     pos: global position of the Duckiebot
        #     angle: global angle of the Duckiebot
        # Outputs:
        #     v: linear veloicy in m/s.
        #     omega: angular velocity, in rad/sec. Right is negative, left is positive.
        
        
        closest_curve_point = env.unwrapped.closest_curve_point
        
        # Find the curve point closest to the agent, and the tangent at that point
        closest_point, closest_tangent = closest_curve_point(pos, angle)
        
       

        iterations = 0
        
        lookup_distance = follow_dist
        multipler = 0.5
        curve_point = None
        
        while iterations < 10:            
            ########
            #
            #TODO 1: Modify follow_point so that it is a function of closest_point, closest_tangent, and lookup_distance
           
            follow_point = closest_point+closest_tangent*lookup_distance
            
            #
            ########
            #follow_point = closest_point
            
            curve_point, _ = closest_curve_point(follow_point, angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= multiplier
        ########
        #
        #TODO 2: Modify omega
        v=0.5
        rot=np.array([[np.cos(angle), 0, np.sin(angle)], [0,1,0], [-np.sin(angle), 0, np.cos(angle)]])
        f_robot=(follow_point-pos).dot(rot)
        alpha=np.arctan2(f_robot[2],f_robot[0])
        omega=-2*v*np.sin(alpha)/lookup_distance
        #
        ########


        return v, omega