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

## Static biplots

Static, publication-ready rendering with matplotlib (see
{doc}`../tutorials/static-biplots`).

```{eval-rst}
.. autofunction:: bipl5.static.plot_static

.. autoclass:: bipl5.static.StaticBiplot
   :members:

.. autofunction:: bipl5.static.add_calibrated_axes

.. autoclass:: bipl5.static.CalibratedAxes
   :members: compute_layout

.. autoclass:: bipl5.static.CalibratedAxis
   :members: from_coordinates

.. autoclass:: bipl5.static.AxisStyle

.. autoclass:: bipl5.static.AxisCalibration
   :members:

.. autofunction:: bipl5.static.axis_calibration

.. autodata:: bipl5.static.plot.ZORDER
```

## Helpers

```{eval-rst}
.. autofunction:: bipl5.colorpal

.. autofunction:: bipl5.symbol_list

.. autofunction:: bipl5.set_print_options
```
