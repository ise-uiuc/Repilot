# Repilot Patch

```

        final int idx = index - count;
        indices[last] = idx;
```

# Developer Patch

```
        indices[last] = index - count;
```

# Context

```
--- bug/Math-56/src/main/java/org/apache/commons/math/util/MultidimensionalCounter.java

+++ fix/Math-56/src/main/java/org/apache/commons/math/util/MultidimensionalCounter.java

@@ -234,12 +234,8 @@

             indices[i] = idx;
         }
 
-        int idx = 1;
-        while (count < index) {
-            count += idx;
-            ++idx;
-        }
-        --idx;
+
+        final int idx = index - count;
         indices[last] = idx;
 
         return indices;
```

# Note

