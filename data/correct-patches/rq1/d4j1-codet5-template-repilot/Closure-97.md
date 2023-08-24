# Repilot Patch

```
result = lvalInt >>> rvalInt & 0xFFFFFFFFL;
```

# Developer Patch

```
          long lvalLong = lvalInt & 0xffffffffL;
          result = lvalLong >>> rvalInt;
```

# Context

```
--- bug/Closure-97/src/com/google/javascript/jscomp/PeepholeFoldConstants.java

+++ fix/Closure-97/src/com/google/javascript/jscomp/PeepholeFoldConstants.java

@@ -695,7 +695,7 @@

           // JavaScript handles zero shifts on signed numbers differently than
           // Java as an Java int can not represent the unsigned 32-bit number
           // where JavaScript can so use a long here.
-          result = lvalInt >>> rvalInt;
+result = lvalInt >>> rvalInt & 0xFFFFFFFFL;
           break;
         default:
           throw new AssertionError("Unknown shift operator: " +
```

# Note

