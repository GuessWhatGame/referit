from referit.models.referit_film_network import FiLM_Oracle


# stupid factory class to create networks
def create_network(config, no_words, reuse=False, device=''):

    network_type = config["type"]

    if network_type == "film":
        return FiLM_Oracle(config, no_words, no_answers=2, reuse=reuse, device=device)
    else:
        assert False, "Invalid network_type: should be: film"


