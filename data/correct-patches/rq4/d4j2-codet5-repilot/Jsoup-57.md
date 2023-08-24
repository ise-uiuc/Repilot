# Repilot Patch

```

                it.remove();
```

# Developer Patch

```
                it.remove();
```

# Context

```
--- bug/Jsoup-57/src/main/java/org/jsoup/nodes/Attributes.java

+++ fix/Jsoup-57/src/main/java/org/jsoup/nodes/Attributes.java

@@ -122,7 +122,8 @@

         for (Iterator<String> it = attributes.keySet().iterator(); it.hasNext(); ) {
             String attrKey = it.next();
             if (attrKey.equalsIgnoreCase(key))
-                attributes.remove(attrKey);
+
+                it.remove();
         }
     }
```

# Note

