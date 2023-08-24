# Repilot Patch

```
        setDataset(dataset);
        //  setDataset(dataset, "Bar Chart");
```

# Developer Patch

```
        setDataset(dataset);
```

# Context

```
--- bug/Chart-12/source/org/jfree/chart/plot/MultiplePiePlot.java

+++ fix/Chart-12/source/org/jfree/chart/plot/MultiplePiePlot.java

@@ -142,7 +142,8 @@

      */
     public MultiplePiePlot(CategoryDataset dataset) {
         super();
-        this.dataset = dataset;
+        setDataset(dataset);
+        //  setDataset(dataset, "Bar Chart");
         PiePlot piePlot = new PiePlot(null);
         this.pieChart = new JFreeChart(piePlot);
         this.pieChart.removeLegend();
```

# Note

