Network error correction
========================

.. automodule:: graphoskop.error_correction
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   fit = gsk.fit_nested_block_model(net, seed = 0)              # node attributes community, community_1, ...
   fit["groups"], fit["description_length"]

   corrected, summary = gsk.measured_network_correction(net, false_negative_rate = 0.1, false_positive_rate = 0.001, seed = 0)
   clean = gsk.extract_backbone(corrected, 0.5, score = "probability", keep_isolates = True)   # maximum marginal posterior network

   screens = gsk.load_network("screens.tsv")                    # columns trials, hits: repeated measurements
   corrected, summary = gsk.measured_network_correction(screens, "trials", "hits", default_measurements = 3)
   summary["false_negative_rate"], summary["false_positive_rate"]   # estimated

   for model in (gsk.gnm_random_graph, gsk.configuration_model, gsk.nested_block_model):   # Fig. 3 of Peixoto (2018)
      corrected, summary = gsk.measured_network_correction(screens, "trials", "hits", default_measurements = 3, model = model)
   corrected, summary = gsk.measured_network_correction(screens, "trials", "hits", default_measurements = 3,
                                                        model = "stochastic_block_model", groups = "lab")   # fixed groups

   scored = gsk.load_network("predictions.tsv")                 # column probability, from another method
   corrected, summary = gsk.uncertain_network_correction(scored, "probability")

.. autofunction:: graphoskop.fit_nested_block_model

.. autofunction:: graphoskop.measured_network_correction

.. autofunction:: graphoskop.uncertain_network_correction
