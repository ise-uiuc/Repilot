# Repilot Patch

```
static int toJavaVersionInt(String version) {
```

# Developer Patch

```
static int toJavaVersionInt(String version) {
```

# Context

```
--- bug/Lang-29/src/main/java/org/apache/commons/lang3/SystemUtils.java

+++ fix/Lang-29/src/main/java/org/apache/commons/lang3/SystemUtils.java

@@ -1669,7 +1669,8 @@

      * 
      * @return the version, for example 131 for Java 1.3.1
      */
-    static float toJavaVersionInt(String version) {
+
+    static int toJavaVersionInt(String version) {
         return toVersionInt(toJavaVersionIntArray(version, JAVA_VERSION_TRIM_SIZE));
     }
```

# Note

