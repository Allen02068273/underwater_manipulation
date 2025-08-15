from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'record_skill'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
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
            'target_transforms = record_skill.target_transforms:main',
            'skill_recorder = record_skill.skill_recorder:main',
            'arm_controller = record_skill.arm_controller:main',
            'perform_skill = record_skill.perform_skill:main'
        ],
    },
)
