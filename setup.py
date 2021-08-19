from setuptools import setup, find_packages

setup(name="jaxtabcorr",
      version="0.0.1.dev1",
      description="Rewriting TabCorr, replacing NumPy with JAX for easy differentiation",
      url="http://github.com/AlanPearl/JaxTabCorr",
      author="Alan Pearl",
      author_email="alanpearl@pitt.edu",
      license="MIT",
      packages=find_packages(),
      python_requires=">=3.8",
      install_requires=[
            "numpy",
            "jax",
            "tabcorr",
            "halotools>=0.7",
            # "colossus",
            # "Corrfunc",
            # "emcee>=3",
            # "mpi4py",
            # "schwimmbad",
            # "corner",
            # "matplotlib",
      ],
      zip_safe=True,
      test_suite="nose.collector",
      tests_require=["nose"],
      )
