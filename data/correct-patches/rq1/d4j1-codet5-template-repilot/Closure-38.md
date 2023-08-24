# Repilot Patch

```

    if ((x < 0 || negativeZero)
        && prev == '-') {
```

# Developer Patch

```
    if ((x < 0 || negativeZero) && prev == '-') {
```

# Context

```
--- bug/Closure-38/src/com/google/javascript/jscomp/CodeConsumer.java

+++ fix/Closure-38/src/com/google/javascript/jscomp/CodeConsumer.java

@@ -242,7 +242,9 @@

     // x--4 (which is a syntax error).
     char prev = getLastChar();
     boolean negativeZero = isNegativeZero(x);
-    if (x < 0 && prev == '-') {
+
+    if ((x < 0 || negativeZero)
+        && prev == '-') {
       add(" ");
     }
```

# Note

