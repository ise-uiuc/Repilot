# Repilot Patch

```
          result =  (lvalInt >>> rvalInt) & 0xffffffffL; //
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
+          result =  (lvalInt >>> rvalInt) & 0xffffffffL; //
           break;
         default:
           throw new AssertionError("Unknown shift operator: " +
```

# Note

When rvalInt is 0 and lvalInt < 0, the 0xffffffffL map will shift the 32 most significant bits to 0 and `result` will be a positive number, unlike the buggy snippet.