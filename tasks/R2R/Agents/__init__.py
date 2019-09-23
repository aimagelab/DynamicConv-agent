from tasks.R2R.Agents.agent import R2RAgent, Oracle, Stop, Random, Dynamic

agents = {'Base': R2RAgent,
          'Oracle': Oracle,
          'Stop': Stop,
          'Random': Random,
          'Dynamic': Dynamic,
          }


def get_agent(name, config):
    assert name in agents.keys(), '%s is not valid agent name' % name
    return agents[name](config)
