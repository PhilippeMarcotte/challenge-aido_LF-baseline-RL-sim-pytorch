import math
import time
import numpy as np

class Controller:
    def __init__(self):
        self.v_bar = 0.23   # Linear velocity
        self.k_d = 3.5     # P gain for d
        self.k_theta = 1     # P gain for theta
        self.d_thres = 0.2615     # Cap for error in d
        self.theta_thres = 0.523     # Maximum desire theta
        self.d_offset = 0.0  # a configurable offset from the lane position

        self.k_Id = 1     # gain for integrator of d
        self.k_Iphi = 0.0    # gain for integrator of phi (phi = theta)
        #TODO: Feedforward was not working, go away with this error source! (Julien)
        self.use_feedforward_part = False
        self.omega_ff = 0
        self.omega_max = 999
        self.omega_min = -999
        self.use_radius_limit = False
        self.min_radius = 0.06

        self.d_ref = 0
        self.phi_ref = 0
        self.object_detected = 0
        self.v_ref_possible = dict()
        self.v_ref_possible["default"] = 1

        self.velocity_to_m_per_s = 1.53
        self.omega_to_rad_per_s = 4.75

        self.cross_track_err = 0
        self.heading_err = 0
        self.cross_track_integral = 0
        self.heading_integral = 0
        self.cross_track_integral_top_cutoff = 0.3
        self.cross_track_integral_bottom_cutoff = -0.3
        self.heading_integral_top_cutoff = 1.2
        self.heading_integral_bottom_cutoff = -1.2

        self.actuator_limits_v = 999.0 # to make sure the limit is not hit before the message is received
        self.actuator_limits_omega = 999.0  # to make sure the limit is not hit before the message is received
        self.omega_max = 999.0  # considering radius limitation and actuator limits   # to make sure the limit is not hit before the message is received

        self.last_ms = None
        self.main_pose_source = "lane_filer"
    
    def computeControlValues(self, d, phi, curvature_ref=0):
        prev_cross_track_err = self.cross_track_err
        prev_heading_err = self.heading_err

        self.cross_track_err = d - self.d_offset
        self.heading_err = phi

        v = self.v_bar

        if v > self.actuator_limits_v:
            v = self.actuator_limits_v

        if math.fabs(self.cross_track_err) > self.d_thres:
            print("inside threshold ")
            self.cross_track_err = self.cross_track_err / math.fabs(self.cross_track_err) * self.d_thres

        currentMillis = int(round(time.time() * 1000))

        if self.last_ms is not None:
            dt = (currentMillis - self.last_ms) / 1000.0
            self.cross_track_integral += self.cross_track_err * dt
            self.heading_integral += self.heading_err * dt

        if self.cross_track_integral > self.cross_track_integral_top_cutoff:
            self.cross_track_integral = self.cross_track_integral_top_cutoff
        if self.cross_track_integral < self.cross_track_integral_bottom_cutoff:
            self.cross_track_integral = self.cross_track_integral_bottom_cutoff

        if self.heading_integral > self.heading_integral_top_cutoff:
            self.heading_integral = self.heading_integral_top_cutoff
        if self.heading_integral < self.heading_integral_bottom_cutoff:
            self.heading_integral = self.heading_integral_bottom_cutoff

        if abs(self.cross_track_err) <= 0.011:  # TODO: replace '<= 0.011' by '< delta_d' (but delta_d might need to be sent by the lane_filter_node.py or even lane_filter.py)
            self.cross_track_integral = 0
        if abs(self.heading_err) <= 0.051:  # TODO: replace '<= 0.051' by '< delta_phi' (but delta_phi might need to be sent by the lane_filter_node.py or even lane_filter.py)
            self.heading_integral = 0
        if np.sign(self.cross_track_err) != np.sign(prev_cross_track_err):  # sign of error changed => error passed zero
            self.cross_track_integral = 0
        if np.sign(self.heading_err) != np.sign(prev_heading_err):  # sign of error changed => error passed zero
            self.heading_integral = 0

        omega_feedforward = v * curvature_ref
        if self.main_pose_source == "lane_filter" and not self.use_feedforward_part:
            omega_feedforward = 0

        # Scale the parameters linear such that their real value is at 0.22m/s TODO do this nice that  * (0.22/self.v_bar)
        omega = self.k_d * (0.22/self.v_bar) * self.cross_track_err + self.k_theta * (0.22/self.v_bar) * self.heading_err
        omega += (omega_feedforward)

        # check if nominal omega satisfies min radius, otherwise constrain it to minimal radius
        if math.fabs(omega) > v / self.min_radius:
            if self.last_ms is not None:
                self.cross_track_integral -= self.cross_track_err * dt
                self.heading_integral -= self.heading_err * dt
            omega = math.copysign(v / self.min_radius, omega)

        #if not self.fsm_state == "SAFE_JOYSTICK_CONTROL":
        # apply integral correction (these should not affect radius, hence checked afterwards)
        omega -= self.k_Id * (0.22/self.v_bar) * self.cross_track_integral
        omega -= self.k_Iphi * (0.22/self.v_bar) * self.heading_integral

        if v == 0:
            omega = 0
        else:
        # check if velocity is large enough such that car can actually execute desired omega
            if v - 0.5 * math.fabs(omega) * 0.1 < 0.065:
                v = 0.065 + 0.5 * math.fabs(omega) * 0.1

        # apply magic conversion factors
        v = v * self.velocity_to_m_per_s
        omega = omega * self.omega_to_rad_per_s

        if omega > self.omega_max: omega = self.omega_max
        if omega < self.omega_min: omega = self.omega_min
        omega += self.omega_ff
        omega = omega
        self.last_ms = currentMillis
        
        return v, omega

    def angle_control_commands(self, dist, angle, controller='P'):
        return self.computeControlValues(dist, angle)
