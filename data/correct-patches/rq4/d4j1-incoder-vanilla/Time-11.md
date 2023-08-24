# Repilot Patch

```
    static ThreadLocal<Boolean> cVerbose = new ThreadLocal<Boolean>() {
        @Override protected Boolean initialValue() { return false; }
    };
```

# Developer Patch

```
    static ThreadLocal<Boolean> cVerbose = new ThreadLocal<Boolean>() {
        protected Boolean initialValue() {
            return Boolean.FALSE;
        }
    };
```

# Context

```
--- bug/Time-11/src/main/java/org/joda/time/tz/ZoneInfoCompiler.java

+++ fix/Time-11/src/main/java/org/joda/time/tz/ZoneInfoCompiler.java

@@ -65,10 +65,9 @@

 
     static Chronology cLenientISO;
 
-    static ThreadLocal<Boolean> cVerbose = new ThreadLocal<Boolean>();
-    static {
-        cVerbose.set(Boolean.FALSE);
-    }
+    static ThreadLocal<Boolean> cVerbose = new ThreadLocal<Boolean>() {
+        @Override protected Boolean initialValue() { return false; }
+    };
 
     /**
      * Gets a flag indicating that verbose logging is required.
```

# Note

