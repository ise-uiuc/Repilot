# Repilot Patch

```
} else if (offsetLocal >= 0) {
```

# Developer Patch

```
        } else if (offsetLocal >= 0) {
```

# Context

```
--- bug/Time-19/src/main/java/org/joda/time/DateTimeZone.java

+++ fix/Time-19/src/main/java/org/joda/time/DateTimeZone.java

@@ -897,7 +897,7 @@

                     return offsetLocal;
                 }
             }
-        } else if (offsetLocal > 0) {
+} else if (offsetLocal >= 0) {
             long prev = previousTransition(instantAdjusted);
             if (prev < instantAdjusted) {
                 int offsetPrev = getOffset(prev);
```

# Note

