# Repilot Patch

```

        final double n1n2prod = n1 * n2;
```

# Developer Patch

```
        final double n1n2prod = n1 * n2;
```

# Context

```
--- bug/Math-30/src/main/java/org/apache/commons/math3/stat/inference/MannWhitneyUTest.java

+++ fix/Math-30/src/main/java/org/apache/commons/math3/stat/inference/MannWhitneyUTest.java

@@ -170,7 +170,8 @@

                                              final int n2)
         throws ConvergenceException, MaxCountExceededException {
 
-        final int n1n2prod = n1 * n2;
+
+        final double n1n2prod = n1 * n2;
 
         // http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U#Normal_approximation
         final double EU = n1n2prod / 2.0;
```

# Note

