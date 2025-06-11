from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'trajectory_generation'

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
            'lte_trajectory_generator = trajectory_generation.lte_trajectory_generator:main',
            'kmp_trajectory_generator = trajectory_generation.kmp_trajectory_generator:main',
            'promp_trajectory_generator = trajectory_generation.promp_trajectory_generator:main',
            'generate_skill_csv = trajectory_generation.generate_skill_csv:main'
        ],
    },
)
