from builtins import staticmethod
import raimitigations.dataprocessing as dp

class MitigationActions:
    """Class for performing mitigation actions on data.
    """

    @staticmethod
    def get_synthesizer(epochs, model):
        """
        Performs sequential feature selection on the data.

        :param epochs: The number of epochs to train the synthesizer for.
        :param model: The model to use for the synthesizer.

        :return: The synthesizer object.        
        :rtype: raimitigations.dataprocessing.Synthesizer
        """

        model_map = { 0:"ctgan", 1:"tvae", 2:"copula", 3:"copula_gan"}

        return dp.Synthesizer(
                    epochs=epochs,
                    model=model_map[model],
                    load_existing=False
                )

    @staticmethod
    def get_rebalancer(type, strategy):
        """
        Performs rebalancing on the data.

        :param type: The type of rebalancing to perform.
        :param strategy: The strategy to use for rebalancing.

        :return: The rebalancer object.
        :rtype: raimitigations.dataprocessing.Rebalance
        """

        args = { 
            'over_sampler': True if (type == 0 or type == 2) else False, 
            'under_sampler': True if (type == 1 or type == 2) else False
        }

        if args['over_sampler']:
            strategy_map = { 0 : "minority", 1 : "not minority", 2 : "not majority", 3 : "all" }
            args['strategy_over'] = strategy_map[strategy]
        
        if args['under_sampler']:
            strategy_map = { 0 : "majority", 1 : "not minority", 2 : "not majority", 3 : "all" }
            args['strategy_under'] = strategy_map[strategy]
        
        print(f'args dict {args}')

        return dp.Rebalance(**args)