from setuptools import setup
import os
from glob import glob

package_name = 'arm_control'

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
            'gamepad_keyboard_control = arm_control.gamepad_keyboard_control:main',
            'master_arm_control = arm_control.master_arm_control:main'
        ],
    },
)
