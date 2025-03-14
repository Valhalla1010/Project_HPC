import cProfile
import pstats
import io
from artificialneuralnetwork import main

def profile():

    p = cProfile.Profile()
    p.enable()

    main()

    p.disable()
    p.dump_stats("profile_result.prof")
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(p, stream=s).sort_stats(sortby)
    ps.print_stats(70)
    print(s.getvalue())


if __name__ == "__main__":
    profile()