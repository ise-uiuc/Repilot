# Repilot Patch

```
indices[last] = index - count;
```

# Developer Patch

```
indices[last] = index - count;
```

# Context

```
--- bug/Math-56/src/main/java/org/apache/commons/math/util/MultidimensionalCounter.java

+++ fix/Math-56/src/main/java/org/apache/commons/math/util/MultidimensionalCounter.java

@@ -234,13 +234,8 @@

             indices[i] = idx;
         }
 
-        int idx = 1;
-        while (count < index) {
-            count += idx;
-            ++idx;
-        }
-        --idx;
-        indices[last] = idx;
+
+        indices[last] = index - count;
 
         return indices;
     }
```

# Note

