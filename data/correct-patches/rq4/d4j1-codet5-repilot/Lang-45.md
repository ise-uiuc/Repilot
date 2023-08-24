# Repilot Patch

```
if (lower > str.length()) {
            lower = str.length();
        }
```

# Developer Patch

```
if (lower > str.length()) {
            lower = str.length();    
        }
```

# Context

```
--- bug/Lang-45/src/java/org/apache/commons/lang/WordUtils.java

+++ fix/Lang-45/src/java/org/apache/commons/lang/WordUtils.java

@@ -613,6 +613,10 @@

 
         // if the lower value is greater than the length of the string,
         // set to the length of the string
+
+        if (lower > str.length()) {
+            lower = str.length();
+        }
         // if the upper value is -1 (i.e. no limit) or is greater
         // than the length of the string, set to the length of the string
         if (upper == -1 || upper > str.length()) {
```

# Note

