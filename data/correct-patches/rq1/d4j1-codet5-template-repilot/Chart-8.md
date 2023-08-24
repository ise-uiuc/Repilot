# Repilot Patch

```
this(time, zone, Locale.getDefault());
```

# Developer Patch

```
        this(time, zone, Locale.getDefault());
```

# Context

```
--- bug/Chart-8/source/org/jfree/data/time/Week.java

+++ fix/Chart-8/source/org/jfree/data/time/Week.java

@@ -172,7 +172,7 @@

      */
     public Week(Date time, TimeZone zone) {
         // defer argument checking...
-        this(time, RegularTimePeriod.DEFAULT_TIME_ZONE, Locale.getDefault());
+this(time, zone, Locale.getDefault());
     }
 
     /**
```

# Note

