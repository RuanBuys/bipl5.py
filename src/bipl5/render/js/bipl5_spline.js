/*
 * Spline-axis interactivity for bipl5 — extracted verbatim from the
 * htmlwidgets::onRender() handler in the R package's insert_spline_js()
 * (biplotEZ_helper.R). Legend clicks toggle whole axes (curve + tick
 * annotations); clicking on an axis drops a prediction tick annotation.
 */
window.bipl5SplineAttach = function (el, x, data) {
  console.log(el);
  var arr1 = new Array(data.p).fill(0);
  el.on('plotly_legendclick', function (dat) {
    if (dat.data[dat.curveNumber].meta[0] === 'data') {
      return;
    }
    if (dat.data[dat.curveNumber].meta[0] === 'density') {
      return;
    }
    if (dat.data[dat.curveNumber].meta[0] === 'axis_pred') {
      return;
    }

    // REMOVE AXES

    var axis = dat.data[dat.curveNumber].legendgroup;
    var num = Number(axis.replace('Ax', ''));

    var indeces = [];
    el.data.forEach(function (item, idx, arr) {
      if (arr[idx].legendgroup === undefined) {
        return;
      }
      if (arr[idx].legendgroup === axis) {
        indeces.push(idx);
      }
      if (arr[idx].customdata === undefined) {
        return;
      }
      if (arr[idx].customdata[0] === axis) {
        indeces.push(idx);
      }
    });

    var old_annotations = el.layout.annotations;
    old_annotations.forEach(function (item, idx, arr) {
      if (arr[idx].customdata === num) {
        old_annotations[idx].visible = !old_annotations[idx].visible;
      }
    });
    var new_annot = { annotations: old_annotations };

    hidden = arr1[num - 1];
    var update = { 'visible': ['legendonly', true][hidden] };
    hidden = [1, 0][hidden];
    arr1[num - 1] = hidden;
    var new_annot = { annotations: old_annotations };

    Plotly.restyle(el.id, update, indeces);

    Plotly.relayout(el.id, new_annot);
    return false;
  });

  //---------------Click on the graph---------------------
  el.on('plotly_click', function (d) {

    // Click on the axes

    console.log(d);
    //el.layout.annotations.push(newAnnotation);
    var NewAnot1 = {
      x: d.points[0].x,
      y: d.points[0].y,
      text: '&#124;',
      showarrow: false,
      meta: 'axis',
      xaxis: 'x',
      yaxis: 'y',
      visible: true,
      textangle: -Math.atan(d.points[0].customdata) * 180 / Math.PI,
      font: {
        size: 8
      },
      customdata: Number(d.points[0].data.legendgroup.replace('Ax', ''))
    };
    var NewAnot2 = {
      x: d.points[0].x,
      y: d.points[0].y,
      text: (d.points[0].hovertext).toString(),
      showarrow: false,
      meta: 'axis',
      xaxis: 'x',
      yaxis: 'y',
      visible: true,
      textangle: -Math.atan(d.points[0].customdata) * 180 / Math.PI,
      yshift: -12 * Math.cos(Math.atan(d.points[0].customdata)),
      xshift: 12 * Math.sin(Math.atan(d.points[0].customdata)),
      font: {
        size: 10
      },
      customdata: Number(d.points[0].data.legendgroup.replace('Ax', ''))
    };

    el.layout.annotations.push(NewAnot1, NewAnot2);
    console.log(el.layout.annotations)
    Plotly.relayout(el.id, { annotations: el.layout.annotations });
  });
};
