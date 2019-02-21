import setuptools

setuptools.setup(
    name='tensorflow_rl',
    version='0.1.17',
    author='ChaKeumgang',
    author_email='chagmgang@gmail.com',
    description='tensorflow-rl: Modular Deep Reinforcement Learning Framework.',
    keywords='tensorflow gym atari tensorflow machine reinforcement learning neural network',
    include_package_data=False,
    packages=setuptools.find_packages(),
    py_modules=['tensorflow_rl'],
    install_requires=[
        'numpy >= 1.13',
        'gym >= 0.10.5',
        'Box2D >= 2.3.2',
        'tensorboardX >= 1.4'
    ],
    extras_require={
        'tf-cpu': [
            'tensorflow == 1.10.0',
        ],
        'tf-gpu': [
            'tensorflow-gpu == 1.10.0',
        ]
    },
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    url='https://github.com/RLOpensource/tensorflow_RL',
)