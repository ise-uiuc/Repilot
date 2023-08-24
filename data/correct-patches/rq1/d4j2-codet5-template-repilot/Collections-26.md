# Repilot Patch

```

    protected Object readResolve() {
```

# Developer Patch

```
    protected Object readResolve() {
```

# Context

```
--- bug/Collections-26/src/main/java/org/apache/commons/collections4/keyvalue/MultiKey.java

+++ fix/Collections-26/src/main/java/org/apache/commons/collections4/keyvalue/MultiKey.java

@@ -274,7 +274,8 @@

      * only stable for the same process).
      * @return the instance with recalculated hash code
      */
-    private Object readResolve() {
+
+    protected Object readResolve() {
         calculateHashCode(keys);
         return this;
     }
```

# Note

