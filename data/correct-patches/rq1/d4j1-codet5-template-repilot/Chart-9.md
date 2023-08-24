# Repilot Patch

```
if (endIndex < 0 || startIndex > endIndex) {
```

# Developer Patch

```
        if ((endIndex < 0)  || (endIndex < startIndex)) {
```

# Context

```
--- bug/Chart-9/source/org/jfree/data/time/TimeSeries.java

+++ fix/Chart-9/source/org/jfree/data/time/TimeSeries.java

@@ -941,7 +941,7 @@

             endIndex = -(endIndex + 1); // this is first item AFTER end period
             endIndex = endIndex - 1;    // so this is last item BEFORE end
         }
-        if (endIndex < 0) {
+if (endIndex < 0 || startIndex > endIndex) {
             emptyRange = true;
         }
         if (emptyRange) {
```

# Note

