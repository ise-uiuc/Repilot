# Repilot Patch

```
        if (endIndex < startIndex || (startIndex < 0 && endIndex < 0)) {
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
+        if (endIndex < startIndex || (startIndex < 0 && endIndex < 0)) {
             emptyRange = true;
         }
         if (emptyRange) {
```

# Note

```
        if (endIndex < startIndex || (startIndex < 0 && endIndex < 0)) {
```
In this snippet, when `endIndex < startIndex` is false, meaning that `startIndex <= endIndex`, the two conditions:
- (startIndex < 0 && endIndex < 0)
- endIndex < 0
are logically equivalent.