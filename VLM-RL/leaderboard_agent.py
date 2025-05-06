import os, gym, carla, xmltodict
import numpy as np
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from clip.clip_rewarded_ppo import CLIPRewardedPPO
from config import CONFIGS
from carla_env.wrappers import *

ROUTE_PATH = os.environ['LEADERBOARD_ROOT'] + '/data/routes_devtest.xml'
with open(ROUTE_PATH, 'r') as file:
    xml_content = file.read()
ROUTE = xmltodict.parse(xml_content)['routes']['route'][0]

def get_entry_point():
    return 'LeaderboardAgent'

class LeaderboardAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        """
        Setup the agent with the configuration file
        :param path_to_conf_file: path to the configuration file
        """

        # host, port = 'localhost', 2000
        # self.client = carla.Client(host, port)
        # self.world  = self.client.get_world()
        # self.route_waypoints = compute_route_waypoints(self.world, self.start_wp, self.end_wp, resolution=1.0)

        self.route_waypoints = ROUTE['waypoints']['position']
        self.current_waypoint_index = 0

        self.CONFIG = CONFIGS["1"]
        # model = CLIPRewardedPPO.load(model_ckpt, config=CONFIG, device="cuda:0", load_clip=False)
        self.observation_space = self._create_observation_space(self.CONFIG)

        # model.inference_only = True
        self.track = Track.MAP

        self.control = carla.VehicleControl()

    def sensors(self):
        """
        Define the sensors for the agent
        :return: list of sensors
        """

        sensors = [
            {'type': 'sensor.camera.semantic_segmentation', 'id': 'SemSeg','x': 0, 'y': 0, 'z': 16,
             'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0, 'width': self.CONFIG['obs_res'][1], 'height': self.CONFIG['obs_res'][0], 'fov': 110.0,},
            {'type': 'sensor.speedometer', 'id': 'Speed'},
            # {'type': 'sensor.opendrive_map', 'id': 'OpenDRIVE', 'reading_frequency': 1},
            {'type': 'sensor.other.gnss', 'id': 'GPS', 'x': 0, 'y': 0, 'z': 0},
            {'type': 'sensor.other.imu', 'id': 'IMU', 'x': 0, 'y': 0, 'z': 0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        ]
        return sensors
    
    def run_step(self, input_data, timestamp):
        self.speed = input_data['Speed'][1]
        self.bev = input_data['SemSeg'][1][:, :, :3]
        self.location = input_data['GPS'][1]
        self.compass = input_data['IMU'][1][6]

        self._try_increment_waypoint()
        return self.control

    def destroy(self):
        pass

    def _create_observation_space(self, CONFIG):
        observation_space = {}
        low, high = [], []
        low.append(-1), high.append(1)
        low.append(0), high.append(1)
        low.append(0), high.append(120)
        observation_space['vehicle_measures'] = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        observation_space['waypoints'] = gym.spaces.Box(low=-50, high=50, shape=(15, 2), dtype=np.float32)
        observation_space['seg_camera'] = gym.spaces.Box(low=0, high=255, shape=(CONFIG['obs_res'][1], CONFIG['obs_res'][0], 3), dtype=np.uint8)

        return gym.spaces.Dict(observation_space)

    def _encode_state(self):
        encoded_state = {}

        # Vehicle measures
        vehicle_measures = []
        vehicle_measures.append(self.control.steer)
        vehicle_measures.append(self.control.throttle)
        vehicle_measures.append(self.speed)
        encoded_state['vehicle_measures'] = vehicle_measures

        # Waypoints
        next_waypoints_state = self.route_waypoints[self.current_waypoint_index: self.current_waypoint_index + 15]
        waypoints = [np.array([float(way['@x']), float(way['@y']), float(way['@z'])]) for way in next_waypoints_state]

        vehicle_location = self.location
        theta = np.deg2rad(90.0 - self.compass)

        relative_waypoints = np.zeros((15, 2))
        for i, w_location in enumerate(waypoints):
            relative_waypoints[i] = get_displacement_vector(vehicle_location, w_location, theta)[:2]
        if len(waypoints) < 15:
            start_index = len(waypoints)
            reference_vector = relative_waypoints[start_index-1] - relative_waypoints[start_index-2]
            for i in range(start_index, 15):
                relative_waypoints[i] = relative_waypoints[i-1] + reference_vector

        encoded_state['waypoints'] = relative_waypoints

        # BEV sec camera
        encoded_state['seg_camera'] = self.bev

        return encoded_state
    
    def _try_increment_waypoint(self):
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            next_waypoint_index = waypoint_index + 1
            way1 = self.route_waypoints[waypoint_index % len(self.route_waypoints)]
            way2 = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            wp_pos = np.array([float(way1['@x']), float(way1['@y']), float(way1['@z'])])
            next_wp_pos = np.array([float(way2['@x']), float(way2['@y']), float(way2['@z'])])

            dir_vec = next_wp_pos - wp_pos
            dir_2d  = dir_vec[:2] / np.linalg.norm(dir_vec[:2])

            to_veh_2d = (self.location - wp_pos)[:2]

            dot = np.dot(dir_2d, to_veh_2d)

            if dot > 0.0:
                waypoint_index += 1
            else:
                break
        self.current_waypoint_index = waypoint_index