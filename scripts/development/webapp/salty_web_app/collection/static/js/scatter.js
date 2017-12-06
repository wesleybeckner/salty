//$(function () {
//
//    // on page load, set the text of the label based the value of the range
//    $('#rangeText').text(rangeValues[$('#rangeInput').val()]);
//
//    // setup an event handler to set the text when the range value is dragged (see event for input) or changed (see event for change)
//    $('#rangeInput').on('input change', function () {
//        $('#rangeText').text(rangeValues[$(this).val()]);
//    });
//
//});

var margin = { top: 50, right: 900, bottom: 50, left: 50 },
    outerWidth = 1400,
    outerHeight = 500,
    width = outerWidth - margin.left - margin.right,
    height = outerHeight - margin.top - margin.bottom;

var x = d3.scale.linear()
    .range([0, width]).nice();

var y = d3.scale.linear()
    .range([height, 0]).nice();

var xCat = "Prediction for alpha of 1.0",
    yCat = "Density (kg/m Experimental)",
    rCat = "Temperature (K)",
    colorCat = "Salt Name";

d3.csv("../static/js/d3_web_data.csv", function(data) {
  data.forEach(function(d) {
    d["Density (kg/m Experimental)"] = +d["Density (kg/m Experimental)"];
    d["Temperature (K)"] = +d["Temperature (K)"];
    d["Pressure (kPa)"] = +d["Pressure (kPa)"];
    d["Prediction for alpha of 1"] = +d["Prediction for alpha of 1.0"];
    d["Prediction for alpha of 0.9"] = +d["Prediction for alpha of 0.9"];
    d["Prediction for alpha of 0.8"] = +d["Prediction for alpha of 0.8"];
    d["Prediction for alpha of 0.7"] = +d["Prediction for alpha of 0.7"];
    d["Prediction for alpha of 0.6"] = +d["Prediction for alpha of 0.6"];
    d["Prediction for alpha of 0.5"] = +d["Prediction for alpha of 0.5"];
    d["Prediction for alpha of 0.4"] = +d["Prediction for alpha of 0.4"];
    d["Prediction for alpha of 0.3"] = +d["Prediction for alpha of 0.3"];
    d["Prediction for alpha of 0.2"] = +d["Prediction for alpha of 0.2"];
    d["Prediction for alpha of 0.1"] = +d["Prediction for alpha of 0.1"];
  });
  var xMax = d3.max(data, function(d) { return d[xCat]; }) * 1.005,
      xMin = d3.min(data, function(d) { return d[xCat]; }),
      yMax = d3.max(data, function(d) { return d[yCat]; }) * 1.005,
      yMin = d3.min(data, function(d) { return d[yCat]; });

  x.domain([xMin, xMax]);
  y.domain([yMin, yMax]);

  var xAxis = d3.svg.axis()
      .scale(x)
      .orient("bottom")
      .tickSize(-height);

  var yAxis = d3.svg.axis()
      .scale(y)
      .orient("left")
      .tickSize(-width);

  var color = d3.scale.category10();


  var tip = d3.tip()
      .attr("class", "d3-tip")
      .offset([-10, 0])
      .html(function(d) {
        return xCat + ": " + d[xCat] + "<br>" + yCat + ": " + d[yCat] + "<br>" + colorCat + ": " + d[colorCat];
      });

  var zoomBeh = d3.behavior.zoom()
      .x(x)
      .y(y)
      .scaleExtent([0, 500])
      .on("zoom", zoom);

  var svg = d3.select("#scatter")
    .append("svg")
      .attr("width", outerWidth)
      .attr("height", outerHeight)
    .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
      .call(zoomBeh);

  svg.call(tip);

  svg.append("rect")
      .attr("width", width)
      .attr("height", height);

  svg.append("g")
      .classed("x axis", true)
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
    .append("text")
      .classed("label", true)
      .attr("x", width)
      .attr("y", margin.bottom - 10)
      .style("text-anchor", "end")
      .text(xCat);

  svg.append("g")
      .classed("y axis", true)
      .call(yAxis)
    .append("text")
      .classed("label", true)
      .attr("transform", "rotate(-90)")
      .attr("y", -margin.left)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text(yCat);

//  var slider = svg.append("svg")
//  var slider = d3.select("#scatter")
//    .append("svg")
//      .attr("width", outerWidth)
//    .append("g")
//      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

//    .append("g")
//    .attr("class", "slider")
//    .attr("width", width)
//    .attr("height", height)
//    .attr("transform", "translate(" + margin.left *10 + "," + margin.top * 5+ ")")

//  slider.append("line")
//    .attr("class", "track")
//    .attr("x1", x.range()[0])
//    .attr("x2", x.range()[1])
//  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
//    .attr("class", "track-inset")
//  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
//    .attr("class", "track-overlay")
//    .call(d3.drag()
//      .on("start.interrupt", function() { slider.interrupt(); })
//      .on("start drag", function() { hue(x.invert(d3.event.x)); }));

  var objects = svg.append("svg")
      .classed("objects", true)
      .attr("width", width)
      .attr("height", height);

  objects.append("svg:line")
      .classed("axisLine hAxisLine", true)
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", width)
      .attr("y2", 0)
      .attr("transform", "translate(0," + height + ")");

  objects.append("svg:line")
      .classed("axisLine vAxisLine", true)
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", height);

  var dot = objects.selectAll(".dot")
      .data(data)
    .enter().append("circle")
      .classed("dot", true)
      .attr("r", function (d) { return 6 * 0.1 * Math.sqrt(d[rCat] / Math.PI); })
      .attr("transform", transform)
      .style("fill", function(d) { return color(d[colorCat]); })
      .attr("data-species", function(d) { return d[colorCat]; })
      .on("mouseover", tip.show)
      .on("mouseout", tip.hide);

  var legend = svg.selectAll(".legend")
      .data(color.domain())
    .enter().append("g")
      .classed("legend", true)
      .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

  legend.append("circle")
      .attr("r", 3.5)
      .attr("cx", width + 20)
      .attr("fill", color)
      .on("click", function(d){
      dot.filter(function () {
		                 return this.dataset.species === d;
				              })
	               .transition().duration(750)
		             .style("opacity", function () {
				                     console.log(this.style.opacity);
						                     return (parseInt(this.style.opacity)) ? 0 : 1;
								                  });

	        });
        // Determine if the data point is visible
//	var active =  test.active ? false : true,
//	  newOpacity = active ? 0 : 1;
//        d3.select("#test").style("opacity", newOpacity);
//	// Update whether or not the elements are active
//	test.active=active;

  legend.append("text")
      .attr("x", width + 46)
      .attr("dy", ".35em")
      .text(function(d) { return d; });

  svg.append("myRange")
  var select = d3.select("#myRange").on("change", change2);

//  svg.append("inds")
//  var select = d3.select("#inds").on("change", change);

  function change2() {
    var sect_value = document.getElementById("myRange").value;
    var sect = "Prediction for alpha of " + sect_value
    console.log(sect)
    xCat = sect;//options[sect.selectedIndex].value;
    xMax = d3.max(data, function(d) { return d[xCat]; });
    xMin = d3.min(data, function(d) { return d[xCat]; });

    zoomBeh.x(x.domain([xMin, xMax])).y(y.domain([yMin, yMax]));

    var svg = d3.select("#scatter").transition();

    svg.select(".x.axis").duration(750).call(xAxis).select(".label").text(xCat);

    objects.selectAll(".dot").transition().duration(1000).attr("transform", transform);
  }

//  function change() {
//    var sect = document.getElementById("inds");
//    console.log(sect)
//    xCat = sect.options[sect.selectedIndex].value;
//    console.log(xCat)
//    xMax = d3.max(data, function(d) { return d[xCat]; });
//    xMin = d3.min(data, function(d) { return d[xCat]; });
//
//    zoomBeh.x(x.domain([xMin, xMax])).y(y.domain([yMin, yMax]));
//
//    var svg = d3.select("#scatter").transition();
//
//    svg.select(".x.axis").duration(750).call(xAxis).select(".label").text(xCat);
//
//    objects.selectAll(".dot").transition().duration(1000).attr("transform", transform);
//  }

  function zoom() {
    svg.select(".x.axis").call(xAxis);
    svg.select(".y.axis").call(yAxis);

    svg.selectAll(".dot")
        .attr("transform", transform);
  }

  function transform(d) {
    return "translate(" + x(d[xCat]) + "," + y(d[yCat]) + ")";
  }
});
