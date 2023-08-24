# Repilot Patch

```
            if (c >= 32 && c <= 126) {
```

# Developer Patch

```
            if (c > 0x1f && c < 0x7f) {
```

# Context

```
--- bug/Closure-73/src/com/google/javascript/jscomp/CodeGenerator.java

+++ fix/Closure-73/src/com/google/javascript/jscomp/CodeGenerator.java

@@ -1042,7 +1042,7 @@

             // No charsetEncoder provided - pass straight latin characters
             // through, and escape the rest.  Doing the explicit character
             // check is measurably faster than using the CharsetEncoder.
-            if (c > 0x1f && c <= 0x7f) {
+            if (c >= 32 && c <= 126) {
               sb.append(c);
             } else {
               // Other characters can be misinterpreted by some js parsers,
```

# Note

0x1f == 31
0x7f == 127