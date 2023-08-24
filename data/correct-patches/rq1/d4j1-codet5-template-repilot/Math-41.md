# Repilot Patch

```
for (int i = begin; i < begin + length; i++) {
```

# Developer Patch

```
                for (int i = begin; i < begin + length; i++) {
```

# Context

```
--- bug/Math-41/src/main/java/org/apache/commons/math/stat/descriptive/moment/Variance.java

+++ fix/Math-41/src/main/java/org/apache/commons/math/stat/descriptive/moment/Variance.java

@@ -517,7 +517,7 @@

                 }
 
                 double sumWts = 0;
-                for (int i = 0; i < weights.length; i++) {
+for (int i = begin; i < begin + length; i++) {
                     sumWts += weights[i];
                 }
```

# Note

