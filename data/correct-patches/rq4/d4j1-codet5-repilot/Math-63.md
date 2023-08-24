# Repilot Patch

```
return equals(x, y, 1);
```

# Developer Patch

```
return equals(x, y, 1);
```

# Context

```
--- bug/Math-63/src/main/java/org/apache/commons/math/util/MathUtils.java

+++ fix/Math-63/src/main/java/org/apache/commons/math/util/MathUtils.java

@@ -414,7 +414,8 @@

      * @return {@code true} if the values are equal.
      */
     public static boolean equals(double x, double y) {
-        return (Double.isNaN(x) && Double.isNaN(y)) || x == y;
+
+        return equals(x, y, 1);
     }
 
     /**
```

# Note

