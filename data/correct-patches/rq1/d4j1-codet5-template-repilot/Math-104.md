# Repilot Patch

```
private static final double DEFAULT_EPSILON = 10e-15;
```

# Developer Patch

```
    private static final double DEFAULT_EPSILON = 10e-15;
```

# Context

```
--- bug/Math-104/src/java/org/apache/commons/math/special/Gamma.java

+++ fix/Math-104/src/java/org/apache/commons/math/special/Gamma.java

@@ -34,7 +34,7 @@

     private static final long serialVersionUID = -6587513359895466954L;
 
     /** Maximum allowed numerical error. */
-    private static final double DEFAULT_EPSILON = 10e-9;
+private static final double DEFAULT_EPSILON = 10e-15;
 
     /** Lanczos coefficients */
     private static double[] lanczos =
```

# Note

