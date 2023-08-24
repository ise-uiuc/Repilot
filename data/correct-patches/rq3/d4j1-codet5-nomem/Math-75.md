# Repilot Patch

```

        return getPct((Comparable<?>) v);
```

# Developer Patch

```
        return getPct((Comparable<?>) v);
```

# Context

```
--- bug/Math-75/src/main/java/org/apache/commons/math/stat/Frequency.java

+++ fix/Math-75/src/main/java/org/apache/commons/math/stat/Frequency.java

@@ -300,7 +300,8 @@

      */
     @Deprecated
     public double getPct(Object v) {
-        return getCumPct((Comparable<?>) v);
+
+        return getPct((Comparable<?>) v);
     }
 
     /**
```

# Note

