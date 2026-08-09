import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'bantala_perception'

here = os.path.abspath(os.path.dirname(__file__))
model_files = [
    os.path.relpath(f, here)
    for f in glob(os.path.join(here, 'bantala_perception', 'models', '*.pt'))
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), model_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='edward',
    maintainer_email='a1victoredward@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'sim_detection = bantala_perception.sim_detection:main',
        ],
    },
)