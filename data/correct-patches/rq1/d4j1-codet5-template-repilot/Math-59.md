# Repilot Patch

```
return (a <= b) ? b : (Float.isNaN(a + b) ? Float.NaN : a);
```

# Developer Patch

```
        return (a <= b) ? b : (Float.isNaN(a + b) ? Float.NaN : a);
```

# Context

```
--- bug/Math-59/src/main/java/org/apache/commons/math/util/FastMath.java

+++ fix/Math-59/src/main/java/org/apache/commons/math/util/FastMath.java

@@ -3479,7 +3479,7 @@

      * @return b if a is lesser or equal to b, a otherwise
      */
     public static float max(final float a, final float b) {
-        return (a <= b) ? b : (Float.isNaN(a + b) ? Float.NaN : b);
+return (a <= b) ? b : (Float.isNaN(a + b) ? Float.NaN : a);
     }
 
     /** Compute the maximum of two values
```

# Note

