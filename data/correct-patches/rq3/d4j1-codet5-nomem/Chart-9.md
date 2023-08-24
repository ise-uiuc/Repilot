# Repilot Patch

```

        if (endIndex < startIndex) {
```

# Developer Patch

```
        if ((endIndex < 0)  || (endIndex < startIndex)) {
```

# Context

```
--- bug/Chart-9/source/org/jfree/data/time/TimeSeries.java

+++ fix/Chart-9/source/org/jfree/data/time/TimeSeries.java

@@ -941,7 +941,8 @@

             endIndex = -(endIndex + 1); // this is first item AFTER end period
             endIndex = endIndex - 1;    // so this is last item BEFORE end
         }
-        if (endIndex < 0) {
+
+        if (endIndex < startIndex) {
             emptyRange = true;
         }
         if (emptyRange) {
```

# Note

