---
layout:     post
title:      "google 地图数模常用接口"
subtitle:   "国外的地图就是详细"
date:       2018-1-23
author:     "LCY"
header-img: "img/default.jpg"
tags:
    - javascript
    - map
    - MCM
    - 数模
---

# 简介

要使用Google地图的接口，必须申请google的key。[Google Maps API获取密钥/身份验证](https://developers.google.com/maps/documentation/javascript/get-api-key)

数模中常用的有在图上标点，绘制热力图，范围图等。[可视化数据：绘制地震图

](https://developers.google.com/maps/documentation/javascript/earthquakes)

## 热力图

```html
<!DOCTYPE html>
<html>
  <head>
    <style>
      /* Always set the map height explicitly to define the size of the div
       * element that contains the map. */
      #map {
        height: 100%;
      }
      /* Optional: Makes the sample page fill the window. */
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
    </style>
  </head>
  <body>
    <div id="map"></div>
    <script>
      var map;
      function initMap() {
        map = new google.maps.Map(document.getElementById('map'), {
          zoom: 2,
          center: {lat: -33.865427, lng: 151.196123},
          mapTypeId: 'terrain'
        });

        // Create a <script> tag and set the USGS URL as the source.
        var script = document.createElement('script');

        // This example uses a local copy of the GeoJSON stored at
        // http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojsonp
        script.src = 'https://developers.google.com/maps/documentation/javascript/examples/json/earthquake_GeoJSONP.js';
        document.getElementsByTagName('head')[0].appendChild(script);

      }

      function eqfeed_callback(results) {
        var heatmapData = [];
        for (var i = 0; i < results.features.length; i++) {
          var coords = results.features[i].geometry.coordinates;
          var latLng = new google.maps.LatLng(coords[1], coords[0]);
          heatmapData.push(latLng);
        }
        var heatmap = new google.maps.visualization.HeatmapLayer({
          data: heatmapData,
          dissipating: false,
          map: map
        });
      }
    </script>
    <script async defer
        src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&libraries=visualization&callback=initMap">
    </script>
  </body>
</html>
```
- eg:

<iframe src="https://developers.google.com/maps/documentation/javascript/examples/full/earthquake_heatmap.jshtml" height="420px" width="100%" allowfullscreen=""></iframe>

## 画圆圈

**官网的脚本是直接调了一个函数去绘制，实际使用的时候一般是自己提供数据**

```html
<!DOCTYPE html>
<html>
  <head>
    <style>
      /* Always set the map height explicitly to define the size of the div
       * element that contains the map. */
      #map {
        height: 100%;
      }
      /* Optional: Makes the sample page fill the window. */
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
    </style>
  </head>
  <body>
    <div id="map"></div>
    <script>
      var map;
      function initMap() {
        map = new google.maps.Map(document.getElementById('map'), {
          zoom: 2,
          center: {lat: -25.363, lng: 131.044},
          mapTypeId: 'terrain'
        });

        // Create a <script> tag and set the USGS URL as the source.
        var script = document.createElement('script');


        // script.src = 'https://developers.google.com/maps/documentation/javascript/examples/json/earthquake_GeoJSONP.js';
        // document.getElementsByTagName('head')[0].appendChild(script);

        map.data.setStyle(function(feature) {
          var magnitude = feature.getProperty('mag');
          return {
            icon: getCircle(magnitude)
          };
        });
        eqfeed_callback();
      }

      function getCircle(magnitude) {
        return {
          path: google.maps.SymbolPath.CIRCLE,
          fillColor: 'red',
          fillOpacity: .2,
          scale: Math.pow(2, magnitude) / 2,
          strokeColor: 'white',
          strokeWeight: .5
        };
      }
      data = [[-25.363, 131.044],[-25.63, 131.044],[-25.363, 131.44]];
      function eqfeed_callback(results) {
        for(var i=0; i<data.length;i++){
          var latLng = new google.maps.LatLng(data[i][0], data[i][1]);
          var marker = new google.maps.Marker({
            map:map,
            icon:getCircle(5),
            position: latLng,
            title:"Hello World!"
          });
        }
      }
      
    </script>
    <script async defer
    src="https://maps.googleapis.com/maps/api/js?key=YOURKEY&callback=initMap">
    </script>
  </body>
</html>
```

官网给的例子：

<iframe src="https://developers.google.com/maps/documentation/javascript/examples/full/earthquake_circles.jshtml" height="420px" width="100%" allowfullscreen=""></iframe>