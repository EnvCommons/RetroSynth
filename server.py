from openreward.environments import Server

from retrosynth import RetroSynth

if __name__ == "__main__":
    server = Server([RetroSynth])
    server.run()
