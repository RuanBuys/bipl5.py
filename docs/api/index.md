# API reference

The user-facing pipeline API. For the internal ordination engine see
{doc}`ordination`.

## The pipeline

```{eval-rst}
.. autofunction:: bipl5.init_biplot

.. autoclass:: bipl5.BiplotSpec
   :members:

.. autoclass:: bipl5.Biplot
   :members:

.. autofunction:: bipl5.format_samples

.. autofunction:: bipl5.score_axes
```

## Widgets and extracted objects

```{eval-rst}
.. autoclass:: bipl5.render.widget.Bipl5Widget
   :members:

.. autoclass:: bipl5.BiplotFit
   :members:

.. autoclass:: bipl5.MdsDisplay

.. autoclass:: bipl5.BiplotData

.. autoclass:: bipl5.FitMeasures
```

## Helpers

```{eval-rst}
.. autofunction:: bipl5.colorpal

.. autofunction:: bipl5.symbol_list

.. autofunction:: bipl5.set_print_options
```
