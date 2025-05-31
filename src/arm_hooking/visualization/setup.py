from setuptools import setup
import os
from glob import glob

package_name = 'visualization'

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
            'visualize_trajectory = visualization.visualize_trajectory:main',
            'apriltag_pose = visualization.apriltag_pose:main',
            'focus_detection = visualization.focus_detection:main'
        ],
    },
)
