from setuptools import setup
import os
from glob import glob

package_name = 'ra_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nolan_allen',
    maintainer_email='nolan_allen@student.uml.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'target_transforms = ra_core.target_transforms:main',
            'apriltag_pose_publisher = ra_core.apriltag_pose_publisher:main',
            'control_node = ra_core.control_node:main',
            'get_brov_cam_pose = ra_core.get_brov_cam_pose:main',
            'inverse_kinematics = ra_core.inverse_kinematics:main'
        ],
    },
)
