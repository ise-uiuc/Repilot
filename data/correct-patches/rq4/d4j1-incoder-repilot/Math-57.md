# Repilot Patch

```
            double sum = 0;
```

# Developer Patch

```
            double sum = 0;
```

# Context

```
--- bug/Math-57/src/main/java/org/apache/commons/math/stat/clustering/KMeansPlusPlusClusterer.java

+++ fix/Math-57/src/main/java/org/apache/commons/math/stat/clustering/KMeansPlusPlusClusterer.java

@@ -172,7 +172,7 @@

         while (resultSet.size() < k) {
             // For each data point x, compute D(x), the distance between x and
             // the nearest center that has already been chosen.
-            int sum = 0;
+            double sum = 0;
             for (int i = 0; i < pointSet.size(); i++) {
                 final T p = pointSet.get(i);
                 final Cluster<T> nearest = getNearestCluster(resultSet, p);
```

# Note

