import unittest

def entrypoint():
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir)
    print(f"Running {suite.countTestCases()} tests")

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    entrypoint()